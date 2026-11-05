"""Geometry and bandwidth model for the scaling_ops.jl work/byte convention.

Rates are decimal bytes/s and FLOP/s. The result is an optimistic model ceiling,
not a prediction: latency, contention, launch cost, and intra-node link costs are
omitted. See hunter_scaling_notes.md for assumptions and hardware sources.
"""
module HunterRoofline

export HunterHardware, benchmark_geometry, operator_work, model_ceiling

Base.@kwdef struct HunterHardware
    hbm_bytes_s::Float64 = 5.3e12
    fp64_flops_s::Float64 = 61.3e12
    fp32_flops_s::Float64 = 122.6e12
    # Do not substitute MI300A matrix-unit throughput for these vector kernels.
    fp16_flops_s::Float64 = Inf # no matrix peak substitution; these FP16 cases are bandwidth-limited
    apus_per_node::Int = 4
    network_bytes_s_per_node::Float64 = 4 * 200e9 / 8
end

function split_grid(token, nprocs, global_dims)
    candidates = NTuple{4,Int}[]
    for i in 1:length(token)-3, j in i+1:length(token)-2, k in j+1:length(token)-1
        parts = (token[1:i], token[i+1:j], token[j+1:k], token[k+1:end])
        any(p -> startswith(p, "0"), parts) && continue
        grid = parse.(Int, parts)
        prod(grid) == nprocs || continue
        all(global_dims .% grid .== 0) || continue
        push!(candidates, grid)
    end
    length(candidates) == 1 || error(
        "Ambiguous/missing process grid '$token'; use a TOML sidecar with numprocs_cart"
    )
    return only(candidates)
end

function benchmark_geometry(filename)
    m = match(r"_(\d+)x(\d+)x(\d+)x(\d+)_(\d+)procs_(\d+)(?:_|\.)", basename(filename))
    isnothing(m) && error("Cannot read lattice/process grid from $filename")
    dims = ntuple(i -> parse(Int, m[i]), 4)
    nprocs = parse(Int, m[5])
    grid = split_grid(m[6], nprocs, dims)
    return (; global_dims=dims, numprocs_cart=grid, nprocs)
end

function operator_work(test_name; gauge_floats=12)
    m = match(r"^(Staggered|Wilson) float(16|32|64)$", test_name)
    isnothing(m) && return nothing
    op, precision = m[1], parse(Int, m[2])
    scalar_bytes = precision ÷ 8
    spinor_reals = op == "Staggered" ? 6 : 24
    flops = op == "Staggered" ? 587 : 1368
    # Exactly the *reported* nominal traffic in scaling_ops.jl, not measured HBM traffic.
    reported_bytes = scalar_bytes * (10spinor_reals + 8gauge_floats)
    # Best-reuse HBM reference: four unique links plus input/output spinors.
    compulsory_bytes = scalar_bytes * (4gauge_floats + 2spinor_reals)
    return (;
        op,
        precision,
        flops,
        reported_bytes,
        compulsory_bytes,
        spinor_bytes=scalar_bytes * spinor_reals,
        intensity=flops / reported_bytes,
    )
end

# MPI Cartesian ranks use the last Cartesian dimension as the fastest index.
function rank_coords(rank, grid)
    coords = zeros(Int, 4)
    for d in 4:-1:1
        rank, coords[d] = divrem(rank, grid[d])
    end
    return Tuple(coords)
end
cart_rank(coords, grid) = foldl((r, d) -> r * grid[d] + coords[d], 1:4; init=0)

function model_ceiling(
    test_name,
    geometry;
    hardware=HunterHardware(),
    gauge_floats=12,
    halo_width=1,
    overlap=false,
    refresh_halo=true,
    hbm_traffic=:benchmark,
    extended_faces=true,
    placement=:contiguous,
)
    work = operator_work(test_name; gauge_floats)
    isnothing(work) && return nothing
    hbm_traffic in (:benchmark, :compulsory) || error("Unknown HBM traffic model")
    placement in (:contiguous, :all_remote) || error("Unknown rank placement")
    hw = hardware
    hw.apus_per_node > 0 || error("apus_per_node must be positive")
    hw.hbm_bytes_s > 0 && hw.network_bytes_s_per_node > 0 ||
        error("Bandwidth must be positive")
    grid = Tuple(geometry.numprocs_cart)
    dims = Tuple(geometry.global_dims)
    n = prod(grid)
    all(dims .% grid .== 0) || error("Global lattice is not divisible by process grid")
    local_dims = dims .÷ grid
    volume = prod(local_dims)
    partitioned = grid .> 1
    face_sites = ntuple(4) do d
        partitioned[d] || return 0
        halo_width * prod(
            if j == d
                1
            else
                local_dims[j] + (extended_faces && j < d && partitioned[j] ? 2halo_width : 0)
            end for j in 1:4
        )
    end
    sent_per_rank = refresh_halo ? 2sum(face_sites) * work.spinor_bytes : 0
    # Sum outgoing bytes per node and phase. Receiving is concurrent/full duplex;
    # summing send+receive would incorrectly halve the link bandwidth.
    node_bytes = zeros(Float64, cld(n, hw.apus_per_node), 4)
    if refresh_halo
        for rank in 0:n-1
            coords = rank_coords(rank, grid)
            node = rank ÷ hw.apus_per_node
            for d in 1:4, side in (-1, 1)
                partitioned[d] || continue
                neighbor = ntuple(
                    j -> j == d ? mod(coords[j] + side, grid[j]) : coords[j], 4
                )
                remote = cart_rank(neighbor, grid) ÷ hw.apus_per_node != node
                if placement == :all_remote || remote
                    node_bytes[node+1, d] += face_sites[d] * work.spinor_bytes
                end
            end
        end
    end
    net_seconds = if overlap
        maximum(sum(node_bytes; dims=2)) / hw.network_bytes_s_per_node
    else
        sum(maximum(node_bytes; dims=1)) / hw.network_bytes_s_per_node
    end
    bytes_per_site = hbm_traffic == :benchmark ? work.reported_bytes : work.compulsory_bytes
    # Packing reads+writes the send payload; unpacking reads+writes the receive payload.
    memory_seconds = (volume * bytes_per_site + 4sent_per_rank) / hw.hbm_bytes_s
    compute_peak = if work.precision == 64
        hw.fp64_flops_s
    elseif work.precision == 32
        hw.fp32_flops_s
    else
        hw.fp16_flops_s
    end
    compute_seconds = volume * work.flops / compute_peak
    local_seconds = max(memory_seconds, compute_seconds)
    seconds = overlap ? max(local_seconds, net_seconds) : local_seconds + net_seconds
    interior = prod(max(0, local_dims[d] - (partitioned[d] ? 2halo_width : 0)) for d in 1:4)
    return (;
        tflops=n * volume * work.flops / seconds / 1e12,
        effective_tb_s=n * volume * work.reported_bytes / seconds / 1e12,
        seconds,
        memory_seconds,
        compute_seconds,
        net_seconds,
        local_dims,
        sent_bytes_per_rank=sent_per_rank,
        max_node_remote_bytes=maximum(sum(node_bytes; dims=2)),
        boundary_fraction=1 - interior / volume,
        intensity=work.intensity,
    )
end

end

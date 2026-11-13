struct CheckpointGaugeBuffer{F,A}
    cpu::F
    staging::A
end

function CheckpointGaugeBuffer(
    U::Gaugefield{B,T,M,GA,N}; staging_bytes=64 * 1024^2
) where {B,T,M,GA,N}
    staging_bytes > 0 || throw(ArgumentError("staging_bytes must be positive"))
    if B === CPU
        return CheckpointGaugeBuffer(convert_field(CPU, U), nothing)
    end

    # Allocate without converting: convert_field would first download the entire
    # GPU layout into a temporary host array.
    cpu = Gaugefield{CPU,T,GA,N}(
        size(U)..., U.β;
        numprocs_cart=U.topology.numprocs_cart,
        halo_width=U.topology.halo_width,
    )
    raw = parent(U.U)
    components = size(raw, 1)
    num_sites = prod(size(raw)[2:5])
    bytes_per_site = components * sizeof(eltype(raw))
    staging_bytes >= bytes_per_site || throw(ArgumentError(
        "staging_bytes must hold at least one packed gauge matrix ($bytes_per_site bytes)"
    ))
    chunk_sites = min(num_sites, staging_bytes ÷ bytes_per_site)
    staging = Array{eltype(raw)}(undef, components, chunk_sites)
    return CheckpointGaugeBuffer(cpu, staging)
end

CheckpointGaugeBuffer(U::AbstractVector; kwargs...) =
    map(u -> CheckpointGaugeBuffer(u; kwargs...), U)

function checkpoint_gauge!(buffer::CheckpointGaugeBuffer, U::Gaugefield{CPU})
    copy!(buffer.cpu, U)
    invalidate_halo!(buffer.cpu)
    return buffer.cpu
end

function checkpoint_gauge!(
    buffer::CheckpointGaugeBuffer, U::Gaugefield{B,T,M,GA,N}
) where {B,T,M,GA,N}
    cpu, staging = buffer.cpu, buffer.staging
    raw = parent(U.U)
    # GPU layout: (component, x, y, z, t, direction), with offset site axes.
    components = size(raw, 1)
    chunk_sites = size(staging, 2)
    sites = CartesianIndices(ntuple(d -> axes(U.U, d + 1), 4))
    num_sites = length(sites)
    bulk = U.topology.bulk_sites

    # Finish producers on all streams before taking the snapshot.
    device_synchronize(B())
    for μ in 1:4
        for first_site in 1:chunk_sites:num_sites
            count = min(chunk_sites, num_sites - first_site + 1)
            source_offset = (μ - 1) * components * num_sites +
                (first_site - 1) * components + 1
            copyto!(staging, 1, raw, source_offset, components * count)
            synchronize(B())
            for i in 1:count
                site = sites[first_site + i - 1]
                if site in bulk
                    cpu[μ, site] = _getindex_mat(Val(N), staging, i, T)
                end
            end
        end
    end

    invalidate_halo!(cpu)
    return cpu
end

function checkpoint_gauge!(buffers::AbstractVector, U::AbstractVector)
    length(buffers) == length(U) || throw(DimensionMismatch("checkpoint field count changed"))
    return map(checkpoint_gauge!, buffers, U)
end

# Preserve the low-level API for callers supplying a CPU field or no workspace.
function checkpoint_gauge!(cpu::Gaugefield, U::Gaugefield)
    convert_field!(cpu, U)
    return cpu
end
checkpoint_gauge!(::Nothing, U::Gaugefield) = convert_field(CPU, U)
checkpoint_gauge!(::Nothing, U::AbstractVector) = map(u -> convert_field(CPU, u), U)


mutable struct Checkpointer{T,TG}
    Ucpu::TG
    checkpoint_dir::String
    checkpoint_every::Int64

    function Checkpointer(U, checkpoint_dir, checkpoint_every)
        if checkpoint_every > 0
            Ucpu = CheckpointGaugeBuffer(U)
            TG = typeof(Ucpu)
            T = JLD2Format
            @level1("- Checkpoints will be created!")
            @level1("|  FORMAT: JLD2")
            @level1("|  DIRECTORY: $(checkpoint_dir)")
            @level1("|  INTERVAL: $(checkpoint_every)")
            @level1("-\n")
        else
            Ucpu = nothing
            TG = Nothing
            T = Nothing
        end

        mpi_amroot(mpi_comm_instance()) && flush(__GlobalLogger[])
        return new{T,TG}(Ucpu, checkpoint_dir, checkpoint_every)
    end
end

function create_checkpoint(
    cp::Checkpointer{T},
    univ,
    updatemethod,
    updatemethod_pt,
    itrj,
    numaccepts,
    numaccepts_t=nothing;
    rank=mpi_myrank()
) where {T}
    T ≡ Nothing && return nothing
    instance = MPI_INSTANCE[]

    if itrj % cp.checkpoint_every == 0
        @level1("|")
        @level1("|  Creating checkpoint")
        filename = joinpath(cp.checkpoint_dir, "checkpoint_$(instance)_$(rank).jld2")
        create_checkpoint(
            T(), cp.Ucpu, univ, updatemethod, updatemethod_pt, itrj, numaccepts, numaccepts_t, filename
        )
        @level1("|  Checkpoint created in $(cp.checkpoint_dir)")
        @level1("|")
        if rank==0
            flush(stdout)
            flush(__GlobalLogger[])
        end
    end

    return nothing
end

function load_checkpoint(parameters; rank=mpi_myrank(), mpi_multi_sim=false, build=false)
    checkpoint_path = parameters.load_checkpoint_path
    @level1("[ Checkpoint loaded from $(checkpoint_path)\n")
    return load_checkpoint(JLD2Format(), parameters; rank, mpi_multi_sim, build)
end

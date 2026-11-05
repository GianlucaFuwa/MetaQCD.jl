# const reqs = fill(Utils.MPI.REQUEST_NULL, 5, 2, 2, 4) # there should be no kernels that need more than 5 fields to be validated
const OVERLAP_DEBUG_TIMING = Val(@load_preference("OVERLAP_DEBUG_TIMING", false))
const OVERLAP_TIMINGS = Dict{Symbol,Float64}()
const OVERLAP_TIMINGS_LOCK = ReentrantLock()

@inline overlap_timing_start() = overlap_timing_start(OVERLAP_DEBUG_TIMING)
@inline overlap_timing_start(::Val{true}) = time_ns()
@inline overlap_timing_start(::Val{false}) = 0

@inline overlap_timing_add!(label::Symbol, t_start) =
    overlap_timing_add!(OVERLAP_DEBUG_TIMING, label, t_start)
@inline function overlap_timing_add!(::Val{true}, label::Symbol, t_start)
    dt = (time_ns() - t_start) * 1e-9
    lock(OVERLAP_TIMINGS_LOCK) do
        OVERLAP_TIMINGS[label] = get(OVERLAP_TIMINGS, label, 0.0) + dt
    end
    return nothing
end
@inline overlap_timing_add!(::Val{false}, ::Symbol, t_start) = nothing

function reset_overlap_timings!()
    lock(OVERLAP_TIMINGS_LOCK) do
        empty!(OVERLAP_TIMINGS)
    end
    return nothing
end

function overlap_timing_snapshot()
    return lock(OVERLAP_TIMINGS_LOCK) do
        Dict(OVERLAP_TIMINGS)
    end
end

# A plan owns only communication metadata/buffers, never the field itself. It is
# cached by that field's HaloValidity and is discarded when a field is deepcopied.
struct HaloExchangePlan{FS,TS,IS,DB,MB,ST,EV,PH}
    from_sites::FS
    to_sites::TS
    index_sites::IS
    sendbufs::DB
    recvbufs::DB
    mpi_sendbufs::MB
    mpi_recvbufs::MB
    neighbors::Vector{Int}
    tags::Vector{Int}
    sendreqs::Utils.MPI.MultiRequest
    recvreqs::Utils.MPI.MultiRequest
    pack_stream::ST
    fill_stream::ST
    producer_event::EV
    fill_event::EV
    done_event::EV
    phases::PH
    staged::Bool
end

@inline function exchange_sites(u, dim, dir, ::Val{edges}, receive=false) where {edges}
    sites = receive ? u.topology.halo_sites[dim][dir] : u.topology.border_sites[dim][dir]
    edges && return sites
    return CartesianIndices(
        ntuple(i -> i == dim ? sites.indices[i] : u.topology.bulk_sites.indices[i], Val(4))
    )
end

function make_halo_plan(u::AbstractMPIField{B}, edges::Val{E}) where {B,E}
    dims = Tuple(findall(>(1), u.topology.numprocs_cart))
    faces = Tuple((dim, dir) for dim in dims for dir in 1:2)
    from_sites = map(f -> exchange_sites(u, f..., edges), faces)
    to_sites = map(f -> exchange_sites(u, f..., edges, true), faces)
    index_sites = map(s -> add_all_indices(u, s), from_sites)
    sendbufs = map(f -> u.sendbuf[2(f[1]-1)+f[2]], faces)
    recvbufs = map(f -> u.recvbuf[2(f[1]-1)+f[2]], faces)
    staged = B != CPU && MPI_IS_GPUAWARE == Val(false)
    # Transfer only the packed face prefix, even though the existing field
    # buffers have capacity for extended corner-propagation slabs.
    send_storage = if staged
        map((b, s) -> Vector{eltype(b)}(undef, length(s)), sendbufs, index_sites)
    else
        sendbufs
    end
    recv_storage = if staged
        map((b, s) -> Vector{eltype(b)}(undef, length(s)), recvbufs, index_sites)
    else
        recvbufs
    end
    mpi_sendbufs = map(
        (b, s) -> Utils.MPI.Buffer(b, length(s), Utils.MPI.Datatype(eltype(b))),
        send_storage,
        index_sites,
    )
    mpi_recvbufs = map(
        (b, s) -> Utils.MPI.Buffer(b, length(s), Utils.MPI.Datatype(eltype(b))),
        recv_storage,
        index_sites,
    )
    neighbors = [
        mpi_cart_shift(u.topology.comm_cart, dim - 1, 1)[dir] for (dim, dir) in faces
    ]
    tags = [2(dim - 1) + dir for (dim, dir) in faces]
    n = length(faces)
    phases = E ? Tuple(i:i+1 for i in 1:2:n) : (1:n,)
    return HaloExchangePlan(
        from_sites,
        to_sites,
        index_sites,
        sendbufs,
        recvbufs,
        mpi_sendbufs,
        mpi_recvbufs,
        neighbors,
        tags,
        Utils.MPI.MultiRequest(n),
        Utils.MPI.MultiRequest(n),
        get_priority_stream(B(), 1),
        get_priority_stream(B(), 2),
        create_stream_event(B()),
        create_stream_event(B()),
        create_stream_event(B()),
        phases,
        staged,
    )
end

function halo_plan(u, edges::Val{E}) where {E}
    v = u.halo_valid
    if v isa HaloValidity
        return get!(v.plans, E) do
            make_halo_plan(u, edges)
        end
    end
    return make_halo_plan(u, edges)
end

@inline halo_pack_field(u) = u
@inline halo_pack_map(u) = Val(false)
@inline halo_source_index(u, index, ::Val{false}) = linear_index(u, index)
@inline function halo_source_index(u, site::CartesianIndex{4}, ::Val{true})
    return linear_index(u, map_to_half(site, u.topology.bulk_sites_padded))
end
@inline function halo_source_index(u, index::CartesianIndex{5}, ::Val{true})
    site = CartesianIndex(index.I[2:5])
    mapped = map_to_half(site, u.topology.bulk_sites_padded)
    return linear_index(u, CartesianIndex((index[1], mapped.I...)))
end

@inline function pack_halo_site!(i, captured)
    field, buffers, sites, lengths, mapping = captured
    face, local_index = get_iterator_index(i, lengths)
    @inbounds buffers[face][local_index] = field.U[halo_source_index(
        field, sites[face][local_index], mapping
    )]
    return nothing
end

function pack_halos!(u::AbstractMPIField{B}, plan, phase) where {B}
    buffers = plan.sendbufs[phase]
    sites = plan.index_sites[phase]
    lengths = length.(sites)
    _parallelfor(
        pack_halo_site!,
        (halo_pack_field(u), buffers, sites, lengths, halo_pack_map(u)),
        1:sum(lengths),
        B,
        256;
        stream=plan.pack_stream,
    )
    return nothing
end

# Receive buffers may still be read by the previous application's fill kernels.
# GPU-aware MPI is not stream ordered: ensure they are reusable before posting.
function exchange_phase!(
    u::AbstractMPIField{B}, plan, phase, tag_base, stream, destination
) where {B}
    isempty(phase) && return nothing
    for i in phase
        mpi_irecv!(
            plan.mpi_recvbufs[i],
            u.topology.comm_cart,
            plan.recvreqs[i];
            source=plan.neighbors[i],
            tag=tag_base + plan.tags[i],
        )
    end
    record_stream_event!(B(), plan.producer_event, stream)
    wait_stream_event!(B(), plan.pack_stream, plan.producer_event)
    t = overlap_timing_start()
    pack_halos!(u, plan, phase)
    overlap_timing_add!(:halo_pack_launch, t)
    # MPI may consume a device buffer immediately. This is the one necessary
    # pack-to-MPI host fence, shared by every face in this phase.
    t = overlap_timing_start()
    synchronize(B(), plan.pack_stream)
    if plan.staged
        for i in phase
            copyto!(
                plan.mpi_sendbufs[i].data,
                1,
                plan.sendbufs[i],
                1,
                length(plan.index_sites[i]),
            )
        end
    end
    for i in phase
        dim_tag = plan.tags[i]
        opposite = isodd(dim_tag) ? dim_tag + 1 : dim_tag - 1
        mpi_isend(
            plan.mpi_sendbufs[i],
            u.topology.comm_cart,
            plan.sendreqs[i];
            dest=plan.neighbors[i],
            tag=tag_base + opposite,
        )
    end
    overlap_timing_add!(:halo_pack_wait_send, t)
    t = overlap_timing_start()
    while true
        completed = Utils.MPI.Waitsome(plan.recvreqs)
        isnothing(completed) && break
        for i in completed
            if plan.staged
                copyto!(
                    plan.recvbufs[i],
                    1,
                    plan.mpi_recvbufs[i].data,
                    1,
                    length(plan.index_sites[i]),
                )
                synchronize(B(), default_stream(B()))
            end
            fill_halo!(u, plan.recvbufs[i], plan.to_sites[i]; stream=plan.fill_stream)
        end
    end
    overlap_timing_add!(:halo_wait_fill, t)
    record_stream_event!(B(), plan.fill_event, plan.fill_stream)
    wait_stream_event!(B(), destination, plan.fill_event)
    wait_stream_event!(B(), plan.pack_stream, plan.fill_event)
    return nothing
end

# This internal operation completes receives and queues their GPU dependencies,
# but deliberately leaves sends outstanding so boundary work can overlap them.
function begin_halo_update!(
    fields::Tuple; do_edges::Val{E}=Val(false), stream, destination=stream
) where {E}
    return ntuple(length(fields)) do i
        u = fields[i]
        halo_is_valid(u, do_edges) && return nothing
        plan = halo_plan(u, do_edges)
        synchronize(get_backend(u)(), plan.fill_stream)
        for phase in plan.phases
            exchange_phase!(u, plan, phase, 8(i - 1), stream, destination)
        end
        validate_halo!(u, do_edges)
        plan
    end
end

finish_halo_sends!(::Nothing) = nothing
function finish_halo_sends!(plan::HaloExchangePlan)
    Utils.MPI.Waitall(plan.sendreqs)
    return nothing
end
function finish_halo_sends!(plans::Tuple)
    foreach(finish_halo_sends!, plans)
    return nothing
end

update_halo!(args...; kwargs...) = nothing
start_halo_update!(args...; kwargs...) = nothing
function update_halo!(
    fields::NTuple{N,AbstractField{B}}; do_edges::Val=Val(true), stream=default_stream(B())
) where {N,B}
    plans = begin_halo_update!(fields; do_edges, stream)
    finish_halo_sends!(plans)
    # Public exchange calls retain their complete-on-return contract.
    synchronize(B(), stream)
    return nothing
end
function start_halo_update!(
    fields::NTuple{N,AbstractField{B}}; do_edges::Val=Val(false), stream=default_stream(B())
) where {N,B}
    update_halo!(fields; do_edges, stream)
    return nothing
end

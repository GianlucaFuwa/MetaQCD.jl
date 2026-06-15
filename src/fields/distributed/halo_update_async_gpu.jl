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

@inline _req_index(i, pn, sr, d, N) = i + (pn - 1) * N + (sr - 1) * N * 2 + (d - 1) * N * 2 * 2

"""
    update_halo!(fields...; do_edges)

Perform a complete halo exchange. Use this function when communication cannot be hidden.
"""
update_halo!(args...; kwargs...) = nothing

function update_halo!(fields::NTuple{N,AbstractMPIField{B}}; kwargs...) where {N,B}
    # reqs = start_halo_update!(fields)
    # finalize_halo_update!(reqs, fields, B)
    synchronize(B(), default_stream(B()))
    start_halo_update!(fields; do_edges=Val(true))
    return nothing
end

"""
    start_halo_update!(fields...)

Start the update of halos or buffers of MPI-parallelized fields and return the Requests.
"""
function start_halo_update!(
    fields::NTuple{N,AbstractMPIField{B}}; do_edges::Val{DO_EDGES}=Val(false)
) where {N,B,DO_EDGES}
    fields_to_update = AbstractMPIField[]
    for u in fields
        !halo_is_valid(u) && push!(fields_to_update, u)
    end

    isempty(fields_to_update) && return nothing

    allocate_commstreams!(B(), fields_to_update)

    if do_edges == Val(false)
        num_pdims = sum(fields[1].topology.numprocs_cart .> 1)
        reqs = fill(Utils.MPI.REQUEST_NULL, length(fields_to_update) * 2 * 2 * num_pdims)
        update_halo_gpu_multi!(reqs, fields_to_update...)
    else
        reqs = fill(Utils.MPI.REQUEST_NULL, length(fields_to_update) * 2 * 2 * 1)
        update_halo_gpu_multi_edges!(reqs, fields_to_update...)
    end

    for u in fields
        validate_halo!(u)
    end

    return nothing
end

function update_halo_gpu_multi!(
    reqs, fields::Vararg{AbstractMPIField{backend},N}
) where {N,backend}
    partitioned_dims = findall(fields[1].topology.numprocs_cart .> 1)
    t_post = overlap_timing_start()
    post_mpi_recvs!(reqs, partitioned_dims, fields...)
    overlap_timing_add!(:halo_post_recv, t_post)

    t_pack = overlap_timing_start()
    sendbufs, streams = launch_packing_kernels!(partitioned_dims, fields...)
    overlap_timing_add!(:halo_pack_launch, t_pack)

    t_send = overlap_timing_start()
    launch_packed_sends!(reqs, sendbufs, streams, partitioned_dims, fields...)
    overlap_timing_add!(:halo_pack_wait_send, t_send)

    filled_streams = falses(size(streams))
    t_waitfill = overlap_timing_start()
    wait_and_fill!(reqs, streams, filled_streams, partitioned_dims, fields...)
    overlap_timing_add!(:halo_wait_fill, t_waitfill)

    t_sync = overlap_timing_start()
    synchronize_filled_streams!(backend, streams, filled_streams)
    overlap_timing_add!(:halo_fill_sync, t_sync)

    return nothing
end

function update_halo_gpu_multi_edges!(
    reqs, fields::Vararg{AbstractMPIField{backend},N}
) where {N,backend}
    partitioned_dims = findall(fields[1].topology.numprocs_cart .> 1)

    for dim in partitioned_dims
        t_post = overlap_timing_start()
        post_mpi_recvs!(reqs, (dim,), fields...)
        overlap_timing_add!(:halo_post_recv, t_post)

        t_pack = overlap_timing_start()
        sendbufs, streams = launch_packing_kernels!((dim,), fields...)
        overlap_timing_add!(:halo_pack_launch, t_pack)

        t_send = overlap_timing_start()
        launch_packed_sends!(reqs, sendbufs, streams, (dim,), fields...)
        overlap_timing_add!(:halo_pack_wait_send, t_send)

        filled_streams = falses(size(streams))
        t_waitfill = overlap_timing_start()
        wait_and_fill!(reqs, streams, filled_streams, (dim,), fields...)
        overlap_timing_add!(:halo_wait_fill, t_waitfill)

        t_sync = overlap_timing_start()
        synchronize_filled_streams!(backend, streams, filled_streams)
        overlap_timing_add!(:halo_fill_sync, t_sync)
    end

    return nothing
end

function launch_packing_kernels!(
    dims, fields::Vararg{AbstractMPIField{backend},N}
) where {backend,N}
    # Pre-allocate send/recv buffers outside loop
    sendbufs = Matrix{Any}(undef, 2N, length(dims))
    streams = Matrix{typeof(default_stream(backend()))}(undef, 2, length(dims))

    for (idim, dim) in enumerate(dims)
        for inbr in 1:2
            stream = get_priority_stream(backend(), inbr + 2(idim-1))
            streams[inbr, idim] = stream
            for ifield in 1:N
                sites_from = fields[ifield].topology.border_sites[dim][inbr]
                idx = (inbr-1)*N + ifield
                sendbufs[idx, idim] = create_sendbuf!(
                    fields[ifield], sites_from, dim, inbr; stream
                )
            end
        end
    end

    return sendbufs, streams
end

function post_mpi_recvs!(
    reqs, dims, fields::Vararg{AbstractMPIField{backend},N}
) where {N,backend}
    topology = fields[1].topology
    comm_cart = topology.comm_cart

    for (idim, dim) in enumerate(dims)
        nbrs = mpi_cart_shift(comm_cart, dim-1, 1)
        for inbr in 1:2
            for ifield in 1:N
                u = fields[ifield]
                tags = 8ifield + (2*(dim-1) + 1), 8ifield + (2*(dim-1) + 2)
                recv_buf = get_recv_buf(u, 2(dim-1) + inbr)

                recv_req = mpi_irecv!(recv_buf, comm_cart; source=nbrs[inbr], tag=tags[inbr])
                reqs[_req_index(ifield, inbr, 1, idim, N)] = recv_req
                reqs[_req_index(ifield, inbr, 2, idim, N)] = Utils.MPI.REQUEST_NULL
            end
        end
    end

    return nothing
end

function launch_packed_sends!(
    reqs, sendbufs, streams, dims, fields::Vararg{AbstractMPIField{backend},N}
) where {N,backend}
    topology = fields[1].topology
    comm_cart = topology.comm_cart

    for (idim, dim) in enumerate(dims)
        nbrs = mpi_cart_shift(comm_cart, dim-1, 1)
        for inbr in 1:2
            synchronize(backend(), streams[inbr, idim])
            for ifield in 1:N
                idx = (inbr - 1) * N + ifield
                tags = 8ifield + (2*(dim-1) + 1), 8ifield + (2*(dim-1) + 2)
                send_req = mpi_isend(
                    sendbufs[idx, idim], comm_cart;
                    dest=nbrs[inbr], tag=tags[mod1(inbr + 1, 2)]
                )
                reqs[_req_index(ifield, inbr, 2, idim, N)] = send_req
            end
        end
    end

    return nothing
end

function wait_and_fill!(
    reqs, streams, filled_streams, dims, fields::Vararg{AbstractMPIField{backend},N}
) where {N,backend}
    active_recvs = [(ifield, inbr, idim, dim) for (idim, dim) in enumerate(dims), inbr in 1:2, ifield in 1:N]
    active_sends = [(ifield, inbr, idim) for (idim, _) in enumerate(dims), inbr in 1:2, ifield in 1:N]

    pending_recvs = length(active_recvs)
    pending_sends = length(active_sends)

    while pending_recvs > 0 || pending_sends > 0
        any_progress = false

        irecv = 1
        while irecv <= length(active_recvs)
            ifield, inbr, idim, dim = active_recvs[irecv]
            recv_idx = _req_index(ifield, inbr, 1, idim, N)
            recv_req = reqs[recv_idx]
            if recv_req == Utils.MPI.REQUEST_NULL || Utils.MPI.Test(recv_req)
                if recv_req != Utils.MPI.REQUEST_NULL
                    u = fields[ifield]
                    recv_buf = get_recv_buf(u, 2*(dim - 1) + inbr)
                    halo_sites = u.topology.halo_sites[dim][inbr]
                    stream = streams[inbr, idim]
                    t_fill = overlap_timing_start()
                    fill_halo!(u, recv_buf, halo_sites; stream=stream)
                    overlap_timing_add!(:halo_fill_launch, t_fill)
                    filled_streams[inbr, idim] = true
                end
                active_recvs[irecv] = active_recvs[end]
                pop!(active_recvs)
                pending_recvs -= 1
                any_progress = true
            else
                irecv += 1
            end
        end

        isend = 1
        while isend <= length(active_sends)
            ifield, inbr, idim = active_sends[isend]
            send_idx = _req_index(ifield, inbr, 2, idim, N)
            send_req = reqs[send_idx]
            if send_req == Utils.MPI.REQUEST_NULL || Utils.MPI.Test(send_req)
                active_sends[isend] = active_sends[end]
                pop!(active_sends)
                pending_sends -= 1
                any_progress = true
            else
                isend += 1
            end
        end

        if !any_progress
            t_wait = overlap_timing_start()
            Utils.MPI.Iprobe(mpi_comm_instance())
            yield()
            overlap_timing_add!(:halo_mpi_wait, t_wait)
        end
    end

    return nothing
end

function synchronize_filled_streams!(::Type{backend}, streams, filled_streams) where backend
    for i in eachindex(filled_streams)
        filled_streams[i] && synchronize(backend(), streams[i])
    end

    return nothing
end

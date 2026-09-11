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
    # synchronize(backend(), default_stream(backend()))

    t_post = overlap_timing_start()
    post_mpi_recvs!(reqs, partitioned_dims, fields...)
    overlap_timing_add!(:halo_post_recv, t_post)

    t_pack = overlap_timing_start()
    sendbufs, send_streams = launch_packing_kernels!(partitioned_dims, fields...)
    overlap_timing_add!(:halo_pack_launch, t_pack)

    t_send = overlap_timing_start()
    launch_packed_sends!(reqs, sendbufs, send_streams, partitioned_dims, fields...)
    overlap_timing_add!(:halo_pack_wait_send, t_send)

    fill_streams = launch_fill_streams(partitioned_dims, fields...)
    filled_streams = falses(size(fill_streams))
    t_waitfill = overlap_timing_start()
    wait_and_fill!(reqs, fill_streams, filled_streams, partitioned_dims, fields...)
    overlap_timing_add!(:halo_wait_fill, t_waitfill)

    t_sync = overlap_timing_start()
    synchronize_filled_streams!(backend, fill_streams, filled_streams)
    overlap_timing_add!(:halo_fill_sync, t_sync)
    finalize_requests!(reqs)
    return nothing
end

function update_halo_gpu_multi_edges!(
    reqs, fields::Vararg{AbstractMPIField{backend},N}
) where {N,backend}
    partitioned_dims = findall(fields[1].topology.numprocs_cart .> 1)
    # synchronize(backend(), default_stream(backend()))

    for dim in partitioned_dims
        t_post = overlap_timing_start()
        post_mpi_recvs!(reqs, (dim,), fields...)
        overlap_timing_add!(:halo_post_recv, t_post)

        t_pack = overlap_timing_start()
        sendbufs, send_streams = launch_packing_kernels!((dim,), fields...)
        overlap_timing_add!(:halo_pack_launch, t_pack)

        t_send = overlap_timing_start()
        launch_packed_sends!(reqs, sendbufs, send_streams, (dim,), fields...)
        overlap_timing_add!(:halo_pack_wait_send, t_send)

        fill_streams = launch_fill_streams((dim,), fields...)
        filled_streams = falses(size(fill_streams))
        t_waitfill = overlap_timing_start()
        wait_and_fill!(reqs, fill_streams, filled_streams, (dim,), fields...)
        overlap_timing_add!(:halo_wait_fill, t_waitfill)

        t_sync = overlap_timing_start()
        synchronize_filled_streams!(backend, fill_streams, filled_streams)
        overlap_timing_add!(:halo_fill_sync, t_sync)
        finalize_requests!(reqs)
    end

    return nothing
end

function launch_packing_kernels!(
    dims, fields::Vararg{AbstractMPIField{backend},N}
) where {backend,N}
    sendbufs = Array{Any}(undef, N, 2, length(dims))
    send_streams = Array{Any}(undef, N, 2, length(dims))

    for (idim, dim) in enumerate(dims)
        for inbr in 1:2, ifield in 1:N
            stream = get_sendstream(backend(), inbr, dim, ifield)
            send_streams[ifield, inbr, idim] = stream
            sites_from = fields[ifield].topology.border_sites[dim][inbr]
            sendbufs[ifield, inbr, idim] = create_sendbuf!(
                fields[ifield], sites_from, dim, inbr; stream
            )
        end
    end

    return sendbufs, send_streams
end

function launch_fill_streams(dims, fields::Vararg{AbstractMPIField{backend},N}) where {backend,N}
    fill_streams = Array{Any}(undef, N, 2, length(dims))

    for (idim, dim) in enumerate(dims)
        for inbr in 1:2, ifield in 1:N
            fill_streams[ifield, inbr, idim] = get_readstream(backend(), inbr, dim, ifield)
        end
    end

    return fill_streams
end

function post_mpi_recvs!(
    reqs, dims, fields::Vararg{AbstractMPIField{backend},N}
) where {N,backend}
    for (idim, dim) in enumerate(dims)
        for inbr in 1:2
            for ifield in 1:N
                topology = fields[ifield].topology
                comm_cart = topology.comm_cart
                nbrs = mpi_cart_shift(comm_cart, dim-1, 1)

                u = fields[ifield]
                tag_base = 8(ifield-1)
                tags = tag_base + (2*(dim-1) + 1), tag_base + (2*(dim-1) + 2)
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
    reqs, sendbufs, send_streams, dims, fields::Vararg{AbstractMPIField{backend},N}
) where {N,backend}
    for (idim, dim) in enumerate(dims)
        for inbr in 1:2, ifield in 1:N
            stream = send_streams[ifield, inbr, idim]
            synchronize(backend(), stream)

            topology = fields[ifield].topology
            comm_cart = topology.comm_cart
            nbrs = mpi_cart_shift(comm_cart, dim-1, 1)
            tag_base = 8(ifield-1)
            tags = tag_base + (2*(dim-1) + 1), tag_base + (2*(dim-1) + 2)
            send_req = mpi_isend(
                sendbufs[ifield, inbr, idim], comm_cart;
                dest=nbrs[inbr], tag=tags[mod1(inbr + 1, 2)]
            )
            reqs[_req_index(ifield, inbr, 2, idim, N)] = send_req
        end
    end

    return nothing
end

function wait_and_fill!(
    reqs, fill_streams, filled_streams, dims, fields::Vararg{AbstractMPIField{backend},N}
) where {N,backend}
    pending_recv = trues(N, 2, length(dims))
    pending_send = trues(N, 2, length(dims))
    pending_recvs = length(pending_recv)
    pending_sends = length(pending_send)

    while pending_recvs > 0 || pending_sends > 0
        any_progress = false

        for (idim, dim) in enumerate(dims)
            for inbr in 1:2, ifield in 1:N
                if pending_recv[ifield, inbr, idim]
                    recv_idx = _req_index(ifield, inbr, 1, idim, N)
                    recv_req = reqs[recv_idx]
                    if recv_req == Utils.MPI.REQUEST_NULL || Utils.MPI.Test(recv_req)
                        if recv_req != Utils.MPI.REQUEST_NULL
                            u = fields[ifield]
                            recv_buf = get_recv_buf(u, 2*(dim - 1) + inbr)
                            halo_sites = u.topology.halo_sites[dim][inbr]
                            stream = fill_streams[ifield, inbr, idim]
                            t_fill = overlap_timing_start()
                            fill_halo!(u, recv_buf, halo_sites; stream=stream)
                            overlap_timing_add!(:halo_fill_launch, t_fill)
                            filled_streams[ifield, inbr, idim] = true
                        end
                        pending_recv[ifield, inbr, idim] = false
                        pending_recvs -= 1
                        any_progress = true
                    end
                end
            end
        end

        for idim in eachindex(dims)
            for inbr in 1:2, ifield in 1:N
                if pending_send[ifield, inbr, idim]
                    send_idx = _req_index(ifield, inbr, 2, idim, N)
                    send_req = reqs[send_idx]
                    if send_req == Utils.MPI.REQUEST_NULL || Utils.MPI.Test(send_req)
                        pending_send[ifield, inbr, idim] = false
                        pending_sends -= 1
                        any_progress = true
                    end
                end
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

@inline function finalize_requests!(reqs)
    # Ensure all non-null requests are fully completed/reaped by MPI
    for i in eachindex(reqs)
        req = reqs[i]
        if req != Utils.MPI.REQUEST_NULL
            Utils.MPI.Wait(req)
            reqs[i] = Utils.MPI.REQUEST_NULL
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

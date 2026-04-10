# const reqs = fill(Utils.MPI.REQUEST_NULL, 5, 2, 2, 4) # there should be no kernels that need more than 5 fields to be validated

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

    # === PHASE 1: Launch all packing kernels concurrently ===
    sendbufs, streams = launch_packing_kernels!(partitioned_dims, fields...)

    # === PHASE 2: Synchronize to ensure packing ===
    for stream in streams
        synchronize(backend(), stream)
    end

    # === PHASE 3: Launch ALL MPI calls (now CPU can overlap with GPU) ===
    launch_mpi_calls!(reqs, sendbufs, partitioned_dims, fields...)

    # === PHASE 4: Wait on recvs and fill halos ===
    wait_and_fill!(reqs, streams, partitioned_dims, fields...)

    # === PHASE 5: Synchronize to ensure filling ===
    for stream in streams
        synchronize(backend(), stream)
    end

    return nothing
end

function update_halo_gpu_multi_edges!(
    reqs, fields::Vararg{AbstractMPIField{backend},N}
) where {N,backend}
    partitioned_dims = findall(fields[1].topology.numprocs_cart .> 1)
    # synchronize(backend(), default_stream(backend()))

    for dim in partitioned_dims
        # === PHASE 1: Launch all packing kernels concurrently ===
        sendbufs, streams = launch_packing_kernels!((dim,), fields...)

        # === PHASE 2: Synchronize to ensure packing ===
        for stream in streams
            synchronize(backend(), stream)
        end

        # === PHASE 3: Launch ALL MPI calls (now CPU can overlap with GPU) ===
        launch_mpi_calls!(reqs, sendbufs, (dim,), fields...)

        # === PHASE 4: Wait on recvs and fill halos ===
        wait_and_fill!(reqs, streams, (dim,), fields...)

        # === PHASE 5: Synchronize to ensure filling ===
        for stream in streams
            synchronize(backend(), stream)
        end
    end

    return nothing
end

function launch_packing_kernels!(
    dims, fields::Vararg{AbstractMPIField{backend},N}
) where {backend,N}
    topology = fields[1].topology
    border_sites = topology.border_sites

    # Pre-allocate send/recv buffers outside loop
    sendbufs = Matrix{Any}(undef, 2N, length(dims))
    streams = Matrix{typeof(default_stream(backend()))}(undef, 2, length(dims))

    for (idim, dim) in enumerate(dims)
        sites_from = border_sites[dim]

        for inbr in 1:2
            stream = get_priority_stream(backend(), inbr + 2(idim-1))
            streams[inbr, idim] = stream
            for ifield in 1:N
                idx = (inbr-1)*N + ifield
                sendbufs[idx, idim] = create_sendbuf!(
                    fields[ifield], sites_from[inbr], dim, inbr; stream
                )
            end
        end
    end

    return sendbufs, streams
end

function launch_mpi_calls!(
    reqs, sendbufs, dims, fields::Vararg{AbstractMPIField{backend},N}
) where {N,backend}
    topology = fields[1].topology
    comm_cart = topology.comm_cart
    req_index(i, pn, sr, d) = i + (pn-1)*N + (sr-1)*N*2 + (d-1)*N*2*2

    for (idim, dim) in enumerate(dims)
        nbrs = mpi_cart_shift(comm_cart, dim-1, 1)
        for inbr in 1:2
            for ifield in 1:N
                idx = (inbr-1)*N + ifield
                u = fields[ifield]
                # tags for this dimension (unique within this call, offset by tag_base)
                tags = 8ifield + (2*(dim-1) + 1), 8ifield + (2*(dim-1) + 2)
                recv_buf = get_recv_buf(u, 2(dim-1) + inbr)

                # Create receive tasks
                recv_req = mpi_irecv!(recv_buf, comm_cart; source=nbrs[inbr], tag=tags[inbr])
                reqs[req_index(ifield, inbr, 1, idim)] = recv_req

                send_req = mpi_isend(sendbufs[idx, idim], comm_cart; dest=nbrs[inbr], tag=tags[mod1(inbr+1, 2)])
                reqs[req_index(ifield, inbr, 2, idim)] = send_req
            end
        end
    end

    return nothing
end

function wait_and_fill!(
    reqs, streams, dims, fields::Vararg{AbstractMPIField{backend},N}
) where {N,backend}
    topology = fields[1].topology
    halo_sites = topology.halo_sites
    req_index(i, pn, sr, d) = i + (pn-1)*N + (sr-1)*N*2 + (d-1)*N*2*2

    sendrecv_ready = fill(false, size(reqs))

    while !(all(sendrecv_ready))
        any_progress = false

        # === ROUND-ROBIN: Test *every* recv request ===
        for (idim, dim) in enumerate(dims)
            for inbr in 1:2
                for i in 1:N
                    recv_idx = req_index(i, inbr, 1, idim)
                    send_idx = req_index(i, inbr, 2, idim)

                    # --- Test recv ---
                    if !sendrecv_ready[recv_idx] && reqs[recv_idx] != Utils.MPI.REQUEST_NULL
                        if Utils.MPI.Test(reqs[recv_idx])  # non-blocking
                            u = fields[i]
                            recv_buf = get_recv_buf(u, 2*(dim-1) + inbr)
                            stream = streams[inbr, idim]
                            fill_halo!(u, recv_buf, halo_sites[dim][inbr]; stream=stream)
                            # Keep any_progress = true while sends are still pending so
                            # we never call yield().  Calling yield() when both ranks
                            # have in-flight sends/recvs can cause a rendezvous deadlock
                            # if neither rank is actively pumping MPI (Test calls
                            # themselves drive MPI progress in non-threaded MPI).
                            sendrecv_ready[recv_idx] = true
                            any_progress = true
                        end
                    end

                    # --- Test send (optional, can be fire-and-forget) ---
                    if !sendrecv_ready[send_idx] && reqs[send_idx] != Utils.MPI.REQUEST_NULL
                        sendrecv_ready[send_idx] = Utils.MPI.Test(reqs[send_idx])
                        any_progress = true
                    end
                end
            end
        end

        # If no progress, yield to MPI progress engine
        if !any_progress
            Utils.MPI.Iprobe(mpi_comm_instance())  # Or your comm
            yield()
        end
    end

    return nothing
end

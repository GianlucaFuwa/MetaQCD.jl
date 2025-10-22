# const reqs = fill(Utils.MPI.REQUEST_NULL, 5, 2, 2, 4) # there should be no kernels that need more than 5 fields to be validated

"""
    update_halo!(fields...; do_edges)

Perform a complete halo exchange. Use this function when communication cannot be hidden.
"""
update_halo!(args...; kwargs...) = nothing

# function update_halo!(fields::NTuple{N,AbstractMPIField{B}}; kwargs...) where {N,B}
#     reqs = start_halo_update!(fields)
#     finalize_halo_update!(reqs, fields, B)
#     return nothing
# end

"""
    start_halo_update!(fields...)

Start the update of halos or buffers of MPI-parallelized fields and return the Requests.
"""
# function start_halo_update!(fields::NTuple{N,AbstractMPIField{B}}; kwargs...) where {N,B}
#     reqs = fill(Utils.MPI.REQUEST_NULL, length(fields), 2, 2, 4)
#     allocate_commstreams!(B(), fields)
#
#     for i in eachindex(fields)
#         halo_is_valid(fields[i]) && continue
#         start_halo_update_single_gpu!(reqs, fields[i], i)
#     end
#
#     finalize_halo_update!(reqs, fields, B)
#     return reqs
# end

function start_halo_update_single_gpu!(reqs, u::AbstractMPIField{backend}, ifield) where {backend}
    topology = u.topology
    comm_cart = topology.comm_cart
    border_sites = topology.border_sites

    for dim in 1:4
        topology.numprocs_cart[dim] == 1 && continue
        prev_nbr, next_nbr = mpi_cart_shift(comm_cart, dim-1, 1)
        prev_sites_from, next_sites_from = border_sites[dim]

        # tags for this dimension (unique within this call, offset by tag_base)
        tag_prev = 8ifield + (2*(dim-1) + 1)
        tag_next = 8ifield + (2*(dim-1) + 2)

        recv_buf_prev = get_recv_buf(u, 2(dim-1) + 1)
        recv_buf_next = get_recv_buf(u, 2(dim-1) + 2)

        # Create receive tasks
        recv_req_prev = mpi_irecv!(recv_buf_prev, comm_cart; source=prev_nbr, tag=tag_prev)
        recv_req_next = mpi_irecv!(recv_buf_next, comm_cart; source=next_nbr, tag=tag_next)
        reqs[ifield, 1, 1, dim] = recv_req_prev
        reqs[ifield, 2, 1, dim] = recv_req_next

        send_stream1 = get_sendstream(backend(), 1, dim, ifield)
        send_stream2 = get_sendstream(backend(), 2, dim, ifield)
        send_buf_prev = create_sendbuf!(u, prev_sites_from, dim, 1; stream=send_stream1)
        send_buf_next = create_sendbuf!(u, next_sites_from, dim, 2; stream=send_stream2)
        synchronize(backend(), send_stream1)
        synchronize(backend(), send_stream2)
        send_req_prev = mpi_isend(send_buf_prev, comm_cart; dest=prev_nbr, tag=tag_next)
        send_req_next = mpi_isend(send_buf_next, comm_cart; dest=next_nbr, tag=tag_prev)
        reqs[ifield, 1, 2, dim] = send_req_prev
        reqs[ifield, 2, 2, dim] = send_req_next
    end

    # validate_halo!(u)
    return nothing
end

"""
finalize_halo_update!(reqs, fields, ::Type{backend})

Wait on all started halo updates in `reqs` to finish.
"""
function finalize_halo_update!(reqs::Array, fields, ::Type{backend}) where {backend}
    sendrecv_ready = fill(false, size(reqs))

    while !(all(sendrecv_ready))
        for dim in 1:4
            for nbr in 1:2
                for i in eachindex(fields)
                    if (reqs[i, nbr, 1, dim] != Utils.MPI.REQUEST_NULL) && (reqs[i, nbr, 2, dim] != Utils.MPI.REQUEST_NULL)
                        # if Utils.MPI.Test(reqs[i, nbr, 1, dim]) && !sendrecv_ready[i, nbr, 1, dim]
                        if !sendrecv_ready[i, nbr, 1, dim]
                            wait(reqs[i, nbr, 1, dim])
                            topology = fields[i].topology
                            halo_sites = topology.halo_sites
                            recv_buf = get_recv_buf(fields[i], 2(dim-1) + nbr)
                            fill_stream = get_readstream(backend(), nbr, dim, i)
                            fill_halo!(fields[i], recv_buf, halo_sites[dim][nbr]; stream=fill_stream)
                            synchronize(backend(), fill_stream)
                            sendrecv_ready[i, nbr, 1, dim] = true
                        end

                        sendrecv_ready[i, nbr, 2, dim] = Utils.MPI.Test(reqs[i, nbr, 2, dim])
                    else
                        sendrecv_ready[i, nbr, 1, dim] = true
                        sendrecv_ready[i, nbr, 2, dim] = true
                    end
                end
            end
        end

        yield()
    end

    return nothing
end

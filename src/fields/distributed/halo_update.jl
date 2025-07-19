"""
    update_halo!(fields...; do_edges)

Perform a complete halo exchange. Use this function when communication cannot be hidden.
"""
update_halo!(args...; kwargs...) = nothing

function update_halo!(
    fields::NTuple{N,AbstractMPIField}; do_edges::Val{DO_EDGES}=Val(true)
) where {N,DO_EDGES}
    # mpi_amroot() && println("start halo update")
    sendrecvreqs = start_halo_update!(fields; do_edges)

    for i in 1:N
        # mpi_amroot() && println("finalize halo update $i")
        finalize_halo_update!(sendrecvreqs[i])
    end

    # mpi_amroot() && println("DONE")
    return nothing
end

"""
    start_halo_update!(fields...; do_edges)

Start the update of halos or buffers of MPI-parallelized fields and return the Requests.
If `do_edges = Val(true)` edges and corners are also transferred via an extended face
propagation scheme. This means that after every dimension the requests have to be
completed and communication cannot be hidden behind computation.
"""
start_halo_update!(args...; kwargs...) = nothing

function start_halo_update!(
    fields::NTuple{N,AbstractMPIField}; do_edges::Val{DO_EDGES}=Val(false)
) where {N,DO_EDGES}
    sendrecv_reqs = ntuple(Val(N)) do i
        if halo_is_valid(fields[i])
            [Utils.MPI.REQUEST_NULL], [Utils.MPI.REQUEST_NULL]
        else
            start_halo_update_single!(fields[i], do_edges)
        end
    end

    return sendrecv_reqs
end

function start_halo_update_single!(
    u::AbstractMPIField{backend}, ::Val{do_edges}=Val(false)
) where {backend,do_edges}
    topology = u.topology
    comm_cart = topology.comm_cart
    halo_sites = topology.halo_sites
    border_sites = topology.border_sites

    all_recv_reqs = Utils.MPI.Request[]
    all_send_reqs = Utils.MPI.Request[]

    for dim in 1:4
        prev_nbr, next_nbr = mpi_cart_shift(comm_cart, dim-1, 1)
        prev_sites_from, next_sites_from = border_sites[dim]
        prev_sites_to, next_sites_to = halo_sites[dim]

        if prev_nbr == next_nbr == mpi_myrank(comm_cart)
            copyto!(u, u, next_sites_to, prev_sites_from)
            copyto!(u, u, prev_sites_to, next_sites_from)
        else
            send_buf_prev = create_sendbuf!(u, prev_sites_from, dim, 1)
            send_buf_next = create_sendbuf!(u, next_sites_from, dim, 2)
            recv_buf_prev = u.halos[2(dim-1) + 1].parent
            recv_buf_next = u.halos[2(dim-1) + 2].parent

            recv_req_prev = mpi_irecv!(recv_buf_prev, comm_cart; source=prev_nbr, tag=1+2(dim-1))
            recv_req_next = mpi_irecv!(recv_buf_next, comm_cart; source=next_nbr, tag=2+2(dim-1))
            send_req_prev = mpi_isend(send_buf_prev, comm_cart; dest=prev_nbr, tag=2+2(dim-1))
            send_req_next = mpi_isend(send_buf_next, comm_cart; dest=next_nbr, tag=1+2(dim-1))

            push!(all_recv_reqs, recv_req_prev)
            push!(all_recv_reqs, recv_req_next)
            push!(all_send_reqs, send_req_prev)
            push!(all_send_reqs, send_req_next)
        end
    end

    validate_halo!(u)
    return all_recv_reqs, all_send_reqs
end

"""
    finalize_halo_update!(reqs...)

Wait on all started halo updates in `reqs` to finish.
"""
finalize_halo_update!(args...) = nothing

function finalize_halo_update!(reqs::Vararg{Tuple{Vector{Utils.MPI.Request},Vector{Utils.MPI.Request}},N}) where N
    for i in 1:N
        finalize_halo_update!(reqs[i])
    end

    return nothing
end

function finalize_halo_update!(reqs::Tuple{Vector{Utils.MPI.Request},Vector{Utils.MPI.Request}})
    recvreqs = reqs[1]
    sendreqs = reqs[2]
    mpi_waitall(recvreqs)
    mpi_waitall(sendreqs)
    return nothing
end


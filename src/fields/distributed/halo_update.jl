"""
    update_halo!(fields...; do_edges)

Perform a complete halo exchange. Use this function when communication cannot be hidden.
"""
update_halo!(args...; kwargs...) = nothing

function update_halo!(
    fields::NTuple{N,AbstractMPIField}; do_edges::Val{DO_EDGES}=Val(true)
) where {N,DO_EDGES}
    # mpi_amroot() && println("start halo update")
    sendrecvtasks = start_halo_update!(fields; do_edges)
    # sendrecvtasks is a Tuple{N} of Tuple{Vector{Task},Vector{Task}} 

    for i in 1:N
        # mpi_amroot() && println("finalize halo update $i")
        finalize_halo_update!(sendrecvtasks[i])
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
    sendrecv_tasks = ntuple(Val(N)) do i
        if halo_is_valid(fields[i])
            [Task(() -> nothing)], [Task(() -> nothing)]
        else
            start_halo_update_single!(fields[i], do_edges)
        end
    end

    return sendrecv_tasks
end

function start_halo_update_single!(
    u::AbstractMPIField{backend}, ::Val{do_edges}=Val(false)
) where {backend,do_edges}
    topology = u.topology
    comm_cart = topology.comm_cart
    halo_sites = topology.halo_sites
    border_sites = topology.border_sites

    all_recv_tasks = Task[]
    all_send_tasks = Task[]

    for dim in 1:4
        prev_nbr, next_nbr = mpi_cart_shift(comm_cart, dim-1, 1)
        prev_sites_from, next_sites_from = border_sites[dim]
        prev_sites_to, next_sites_to = halo_sites[dim]

        # If edges matter, wait for previous dimension
        if do_edges && dim > 1
            # Wait for all tasks from previous dimensions
            for task in all_recv_tasks
                cooperative_wait(task)
            end

            for task in all_send_tasks
                cooperative_wait(task)
            end
            # Clear completed tasks
            empty!(all_recv_tasks)
            empty!(all_send_tasks)
        end

        if prev_nbr == next_nbr == mpi_myrank(comm_cart)
            copyto!(u, u, next_sites_to, prev_sites_from)
            copyto!(u, u, prev_sites_to, next_sites_from)
        else
            send_buf_prev = create_sendbuf!(u, prev_sites_from, dim, 1)
            send_buf_next = create_sendbuf!(u, next_sites_from, dim, 2)
            recv_buf_prev = u.halos[2(dim-1) + 1].parent
            recv_buf_next = u.halos[2(dim-1) + 2].parent

            # Start receives first (these must be started on main thread)
            recv_req_prev = mpi_irecv!(recv_buf_prev, comm_cart; source=prev_nbr, tag=1+2(dim-1))
            recv_req_next = mpi_irecv!(recv_buf_next, comm_cart; source=next_nbr, tag=2+2(dim-1))

            # Create receive tasks
            recv_task = Base.Threads.@spawn begin
                KA.priority!(backend(), :high)
                try
                    wait(recv_req_prev)
                catch err
                    error(err)
                end
                try
                    wait(recv_req_next)
                catch err
                    error(err)
                end
                KA.synchronize(backend())
            end

            push!(all_recv_tasks, recv_task)

            send_task = Base.Threads.@spawn begin
                send_req_prev = mpi_isend(send_buf_prev, comm_cart; dest=prev_nbr, tag=2+2(dim-1))
                send_req_next = mpi_isend(send_buf_next, comm_cart; dest=next_nbr, tag=1+2(dim-1))
                try
                    wait(send_req_prev)
                catch err
                    error(err)
                end
                try
                    wait(send_req_next)
                catch err
                    error(err)
                end
            end

            push!(all_send_tasks, send_task)
        end
    end

    validate_halo!(u)
    return all_recv_tasks, all_send_tasks
end

"""
    finalize_halo_update!(reqs...)

Wait on all started halo updates in `reqs` to finish.
"""
finalize_halo_update!(args...) = nothing

function finalize_halo_update!(reqs::Vararg{Tuple{Vector{Task},Vector{Task}},N}) where N
    for i in 1:N
        finalize_halo_update!(reqs[i])
    end

    return nothing
end

function finalize_halo_update!(tasks::Tuple{Vector{Task},Vector{Task}})
    recvtasks = tasks[1]
    sendtasks = tasks[2]

    for recvtask in recvtasks
        cooperative_wait(recvtask)
    end

    for sendtask in sendtasks
        cooperative_wait(sendtask)
    end

    return nothing
end


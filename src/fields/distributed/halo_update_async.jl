"""
    update_halo!(fields...; do_edges)

Perform a complete halo exchange. Use this function when communication cannot be hidden.
"""
function update_halo!(
    fields::NTuple{N,AbstractMPIField}; do_edges::Val{DO_EDGES}=Val(false)
) where {N,DO_EDGES}
    sendrecvtasks = start_halo_update!(fields; do_edges)
    # sendrecvtasks is a Tuple{N} of Tuple{Vector{Task},Vector{Task}} 

    for i in 1:N
        finalize_halo_update!(sendrecvtasks[i])
    end

    return nothing
end

"""
    start_halo_update!(fields...; do_edges)

Start the update of halos or buffers of MPI-parallelized fields and return the Requests.
If `do_edges = Val(true)` edges and corners are also transferred via an extended face
propagation scheme. This means that after every dimension the requests have to be
completed and communication cannot be hidden behind computation.
"""
start_halo_update!(args...; kwargs...) = ()

function start_halo_update!(
    fields::NTuple{N,AbstractMPIField}; do_edges::Val{DO_EDGES}=Val(false)
) where {N,DO_EDGES}
    sendrecv_tasks = ntuple(Val(N)) do i
        if halo_is_valid(fields[i])
            [Task(() -> nothing)], [Task(() -> nothing)]
        else
            start_halo_update_single!(fields[i], do_edges; tag_base=i)
        end
    end

    return sendrecv_tasks
end

# function start_halo_update_single!(
#     u::AbstractMPIField{backend}, ::Val{do_edges}=Val(false); tag_base::Int=1
# ) where {backend,do_edges}
#     topology = u.topology
#     comm_cart = topology.comm_cart
#     halo_sites = topology.halo_sites
#     border_sites = topology.border_sites
#     convert_fun = if MPI_IS_GPUAWARE == Val(false)
#         array_type(backend) 
#     else
#         identity
#     end
#
#     all_recv_tasks = Vector{Task}(undef, 2sum(topology.numprocs_cart .> 1))
#     all_send_tasks = Vector{Task}(undef, 2sum(topology.numprocs_cart .> 1))
#     itask = 0
#
#     for dim in 1:4
#         prev_nbr, next_nbr = mpi_cart_shift(comm_cart, dim-1, 1)
#         prev_sites_from, next_sites_from = border_sites[dim]
#         prev_sites_to, next_sites_to = halo_sites[dim]
#
#         # tags for this dimension (unique within this call, offset by tag_base)
#         tag_prev = 8tag_base + (2*(dim-1) + 1)
#         tag_next = 8tag_base + (2*(dim-1) + 2)
#
#         if topology.numprocs_cart[dim] != 1
#             itask += 1
#             recv_buf_prev = get_recv_buf(u, 2(dim-1) + 1)
#             recv_buf_next = get_recv_buf(u, 2(dim-1) + 2)
#
#             # Create receive tasks
#             recv_req_prev = mpi_irecv!(recv_buf_prev, comm_cart; source=prev_nbr, tag=tag_prev)
#             recv_req_next = mpi_irecv!(recv_buf_next, comm_cart; source=next_nbr, tag=tag_next)
#             recv_task_prev = @task begin
#                 priority!(backend(), :high)
#                 wait(recv_req_prev)
#                 fill_halo!(u, convert_fun(recv_buf_prev), prev_sites_to)
#                 synchronize(backend())
#             end
#
#             recv_task_next = @task begin
#                 priority!(backend(), :high)
#                 wait(recv_req_next)
#                 fill_halo!(u, convert_fun(recv_buf_next), next_sites_to)
#                 synchronize(backend())
#             end
#
#             schedule(recv_task_prev)
#             schedule(recv_task_next)
#             all_recv_tasks[1 + 2(itask-1)] = recv_task_prev
#             all_recv_tasks[2 + 2(itask-1)] = recv_task_next
#
#             send_task_prev = @task begin
#                 priority!(backend(), :high)
#                 send_buf_prev = create_sendbuf!(u, prev_sites_from, dim, 1)
#                 synchronize(backend())
#                 send_req_prev = mpi_isend(send_buf_prev, comm_cart; dest=prev_nbr, tag=tag_next)
#                 wait(send_req_prev)
#             end
#
#             send_task_next = @task begin
#                 priority!(backend(), :high)
#                 send_buf_next = create_sendbuf!(u, next_sites_from, dim, 2)
#                 synchronize(backend())
#                 send_req_next = mpi_isend(send_buf_next, comm_cart; dest=next_nbr, tag=tag_prev)
#                 wait(send_req_next)
#             end
#
#             schedule(send_task_prev)
#             schedule(send_task_next)
#             all_send_tasks[1 + 2(itask-1)] = send_task_prev
#             all_send_tasks[2 + 2(itask-1)] = send_task_next
#         end
#     end
#
#     # validate_halo!(u)
#     return all_recv_tasks, all_send_tasks
# end

"""
    finalize_halo_update!(reqs...)

Wait on all started halo updates in `reqs` to finish.
"""
finalize_halo_update!(::Tuple{}) = nothing

function finalize_halo_update!(reqs::NTuple{N,Tuple{Vector{Task},Vector{Task}}}, args...) where N
    for i in 1:N
        finalize_halo_update!(reqs[i])
    end

    return nothing
end

function finalize_halo_update!(tasks::Tuple{Vector{Task},Vector{Task}})
    recvtasks = tasks[1]
    sendtasks = tasks[2]

    @sync for i in eachindex(recvtasks, sendtasks)
        @async begin
            cooperative_wait(recvtasks[i])
            cooperative_wait(sendtasks[i])
        end
    end

    return nothing
end

function cooperative_wait(task::Task)
    while !Base.istaskdone(task)
        try
            Utils.MPI.Iprobe(mpi_comm())
        catch e
            println("error in iprobe: $e")
            rethrow()
        end
        yield()
    end

    wait(task)
    return nothing
end

function start_halo_update_single!(
    u::AbstractMPIField{backend}, ::Val{do_edges}=Val(false); tag_base::Int=1
) where {backend,do_edges}
    topology = u.topology
    comm_cart = topology.comm_cart
    halo_sites = topology.halo_sites
    border_sites = topology.border_sites
    convert_fun = if MPI_IS_GPUAWARE == Val(false)
        array_type(backend) 
    else
        identity
    end

    all_recv_tasks = Vector{Task}(undef, sum(topology.numprocs_cart .> 1))
    all_send_tasks = Vector{Task}(undef, sum(topology.numprocs_cart .> 1))
    itask = 0

    for dim in 1:4
        prev_nbr, next_nbr = mpi_cart_shift(comm_cart, dim-1, 1)
        prev_sites_from, next_sites_from = border_sites[dim]
        prev_sites_to, next_sites_to = halo_sites[dim]

        # tags for this dimension (unique within this call, offset by tag_base)
        tag_prev = 8tag_base + (2*(dim-1) + 1)
        tag_next = 8tag_base + (2*(dim-1) + 2)

        if topology.numprocs_cart[dim] != 1
            itask += 1
            recv_buf_prev = get_recv_buf(u, 2(dim-1) + 1)
            recv_buf_next = get_recv_buf(u, 2(dim-1) + 2)

            # Create receive tasks
            recv_req_prev = mpi_irecv!(recv_buf_prev, comm_cart; source=prev_nbr, tag=tag_prev)
            recv_req_next = mpi_irecv!(recv_buf_next, comm_cart; source=next_nbr, tag=tag_next)
            recv_task = @task begin
                priority!(backend(), :high)
                wait(recv_req_prev)
                wait(recv_req_next)
                fill_halo!(u, convert_fun(recv_buf_prev), prev_sites_to)
                fill_halo!(u, convert_fun(recv_buf_next), next_sites_to)
                synchronize(backend())
            end

            schedule(recv_task)
            all_recv_tasks[itask] = recv_task

            send_task = @task begin
                priority!(backend(), :high)
                send_buf_prev = create_sendbuf!(u, prev_sites_from, dim, 1)
                send_buf_next = create_sendbuf!(u, next_sites_from, dim, 2)
                synchronize(backend())
                send_req_prev = mpi_isend(send_buf_prev, comm_cart; dest=prev_nbr, tag=tag_next)
                send_req_next = mpi_isend(send_buf_next, comm_cart; dest=next_nbr, tag=tag_prev)
                wait(send_req_prev)
                wait(send_req_next)
            end

            schedule(send_task)
            all_send_tasks[itask] = send_task
        end
    end

    # validate_halo!(u)
    return all_recv_tasks, all_send_tasks
end

using MPI

struct test
    a::Vector{Float64}
end

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if rank == 0
        global_array = Vector{test}(undef, nprocs)

        for i in 1:nprocs
            global_array[i] = test([i, 2i])
        end
        local_array = MPI.Scatter(global_array, test, comm)
    end
    MPI.Barrier(comm)

    # Fill the local portion of the vector with the process rank
    local_array.a += rank

    # Gather all local arrays from different processes into the global vector
    global_array = MPI.Allgather(local_array, comm)

    return global_array
end

global_array = main()

# Print the global vector
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    println("Global vector: ", global_array)
end

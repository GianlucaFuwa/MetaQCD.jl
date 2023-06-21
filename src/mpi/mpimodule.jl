module MPIModule
    using MPI

    MPI.Init()

    const comm = MPI.COMM_WORLD
    const myrank = MPI.Comm_rank(comm)
    const nprocs = MPI.Comm_size(comm)

    get_comm() = comm
    get_myrank() = myrank
    get_nprocs() = nprocs

    function println_rank0(jj...)
        if myrank == 0
            println(jj...)
        end

        return nothing
    end

end

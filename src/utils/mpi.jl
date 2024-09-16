"""
    update_halo!(U)

Update the halos or buffers of an MPI-parallelized field.
"""
function update_halo! end # INFO: Is declared here, so both Fields and MetaIO can access it

"""
    mpi_init()

Check whether MPI has been initialized and if not, initialize it.
"""
mpi_init() = (MPI.Initialized() || MPI.Init())

@inline function mpi_comm()
    mpi_init()
    return MPI.COMM_WORLD
end

@inline function mpi_size()
    mpi_init()
    return MPI.Comm_size(mpi_comm())
end

@inline function mpi_parallel()
    mpi_init()
    return mpi_size() > 1
end

@inline function mpi_myrank()
    mpi_init()
    return MPI.Comm_rank(mpi_comm())
end

@inline function mpi_amroot()
    mpi_init()
    return mpi_myrank() == 0
end

@inline function mpi_barrier()
    mpi_init()
    return MPI.Barrier(mpi_comm())
end

@inline function mpi_barrier(comm)
    return MPI.Barrier(comm)
end

@inline function mpi_cart_create(args...; kwargs...)
    mpi_init()
    return MPI.Cart_create(mpi_comm(), args...; kwargs...)
end

@inline function mpi_cart_shift(comm_cart, dir, disp)
    return MPI.Cart_shift(comm_cart, dir, disp)
end

@inline function mpi_cart_coords(comm_cart, rank)
    return MPI.Cart_coords(comm_cart, rank)
end

@inline function mpi_multirequest(n)
    return MPI.MultiRequest(n)
end

@inline function mpi_send(args...; kwargs...)
    return MPI.Send(args...; kwargs...)
end

@inline function mpi_isend(args...; kwargs...)
    return MPI.Isend(args...; kwargs...)
end

@inline function mpi_recv(args...; kwargs...)
    return MPI.Recv(args...; kwargs...)
end

@inline function mpi_irecv!(args...; kwargs...)
    return MPI.Irecv!(args...; kwargs...)
end

@inline function mpi_waitall(args...)
    return MPI.Waitall(args...)
end

@inline function mpi_allreduce(sendbuf, op, comm)
    return MPI.Allreduce(sendbuf, op, comm)
end

@inline function mpi_allgather(sendbuf, comm)
    return MPI.Allgather(sendbuf, comm)
end

@inline function mpi_bcast!(buff, comm; root=0)
    return MPI.Bcast!(buff, comm, root=root)
end

@inline function mpi_bcast_isbits(obj, comm=mpi_comm(); root=0)
    return MPI.bcast(obj, comm, root=root)
end

@inline function mpi_write_at(fp, offset, data)
    return MPI.File.write_at(fp, offset, data)
end

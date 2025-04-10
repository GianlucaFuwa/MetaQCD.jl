"""
    update_halo!(U)

Update the halos or buffers of an MPI-parallelized field.
"""
update_halo!(args...) = nothing # INFO: Is declared here, so both Fields and MetaIO can access it

const MPI_INSTANCE_INITIALIZED = Base.RefValue{Bool}(false)
const MPI_COMM_WORLD = Base.RefValue{MPI.Comm}()
const MPI_COMM_INSTANCE = Base.RefValue{MPI.Comm}()
const MPI_COMM_ROOT = Base.RefValue{MPI.Comm}()
const MPI_WORLD_SIZE = Base.RefValue{Int64}()
const MPI_INSTANCE_SIZE = Base.RefValue{Int64}()
const MPI_NUMINSTANCES = Base.RefValue{Int64}(1)
const MPI_INSTANCE = Base.RefValue{Int64}(0)

"""
    mpi_init()

Check whether MPI has been initialized and if not, initialize it.
"""
function mpi_init()
    if MPI.Initialized()
        return nothing
    else
        MPI.Init(finalize_atexit=true)
        MPI_COMM_WORLD[] = MPI.COMM_WORLD
        MPI_COMM_INSTANCE[] = MPI.COMM_WORLD
        MPI_COMM_ROOT[] = MPI.COMM_WORLD
        MPI_WORLD_SIZE[] = mpi_size(MPI.COMM_WORLD)
    end
end

@inline function mpi_comm()
    mpi_init()
    return MPI.COMM_WORLD
end

@inline function mpi_comm_instance()
    mpi_init()
    return MPI_COMM_INSTANCE[]
end

@inline function mpi_comm_root()
    mpi_init()
    return MPI_COMM_ROOT[]
end

@inline function mpi_size(comm=mpi_comm())
    return MPI.Comm_size(comm)
end

@inline function mpi_parallel(comm=mpi_comm())
    return mpi_size(comm) > 1
end

@inline function mpi_myrank(comm=mpi_comm())
    return MPI.Comm_rank(comm)
end

@inline function mpi_amroot(comm=mpi_comm())
    return mpi_myrank(comm) == 0
end

@inline function mpi_split(comm=mpi_comm(); color=0, key=0)
    comm_split = MPI.Comm_split(comm, color, key)
    MPI_INSTANCE_INITIALIZED[] = true
    MPI_COMM_INSTANCE[] = comm_split
    MPI_INSTANCE[] = color

    comm_root = MPI.Comm_split(comm, MPI.Comm_rank(comm_split), 0)
    MPI_COMM_ROOT[] = comm_root
    return comm_split
end

@inline function mpi_barrier(comm=mpi_comm())
    mpi_init()
    return MPI.Barrier(comm)
end

@inline function mpi_cart_create(comm, args...; kwargs...)
    return MPI.Cart_create(comm, args...; kwargs...)
end

@inline function mpi_cart_shift(comm_cart, dir, disp)
    return MPI.Cart_shift(comm_cart, dir, disp)
end

@inline function mpi_cart_coords(comm_cart)
    return MPI.Cart_coords(comm_cart)
end

@inline function mpi_multirequest(n)
    return MPI.MultiRequest(n)
end

@inline function mpi_send(args...; kwargs...)
    return MPI.Send(args...; kwargs...)
end

@inline function mpi_ssend(args...; kwargs...)
    return MPI.send(args...; kwargs...)
end

@inline function mpi_isend(args...; kwargs...)
    return MPI.Isend(args...; kwargs...)
end

@inline function mpi_recv(args...; kwargs...)
    return MPI.Recv(args...; kwargs...)
end

@inline function mpi_srecv(args...; kwargs...)
    return MPI.recv(args...; kwargs...)
end

@inline function mpi_irecv!(args...; kwargs...)
    return MPI.Irecv!(args...; kwargs...)
end

@inline function mpi_waitall(args...)
    return MPI.Waitall(args...)
end

@inline function mpi_allreduce(sendbuf::T, op, comm) where {T}
    return MPI.Allreduce(sendbuf::T, op, comm)
end

@inline function mpi_allgather(sendbuf::T, comm) where {T}
    return MPI.Allgather(sendbuf::T, comm)
end

@inline function mpi_bcast!(buff, comm; root=0)
    return MPI.Bcast!(buff, comm, root=root)
end

@inline function mpi_bcast(obj::T, comm; root=0) where {T}
    return MPI.bcast(obj::T, comm, root=root)
end

@inline function mpi_bcast_isbits(obj::T, comm=mpi_comm(); root=0) where {T}
    return MPI.bcast(obj::T, comm, root=root)
end

@inline function mpi_write_at(fp, offset, data)
    return MPI.File.write_at(fp, offset, data)
end

const MPI_INSTANCE_INITIALIZED = Base.RefValue{Bool}(false)
const MPI_COMM_WORLD = Base.RefValue{MPI.Comm}()
const MPI_COMM_INSTANCE = Base.RefValue{MPI.Comm}()
const MPI_COMM_SHARED = Base.RefValue{MPI.Comm}()
const MPI_WORLD_SIZE = Base.RefValue{Int64}()
const MPI_INSTANCE_SIZE = Base.RefValue{Int64}()
const MPI_NUMINSTANCES = Base.RefValue{Int64}(1)
const MPI_INSTANCE = Base.RefValue{Int64}(0)
const MPI_IS_GPUAWARE = Val(@load_preference("MPI_IS_GPUAWARE", false))

"""
    mpi_init()

Check whether MPI has been initialized and if not, initialize it.
"""
function mpi_init()
    if MPI.Initialized()
        return nothing
    else
        MPI.Init(; threadlevel=:multiple, finalize_atexit=true)
        MPI_COMM_WORLD[] = MPI.COMM_WORLD
        MPI_COMM_INSTANCE[] = MPI.COMM_WORLD
        MPI_COMM_SHARED[] = MPI.COMM_WORLD
        MPI_WORLD_SIZE[] = mpi_size(MPI.COMM_WORLD)
        return nothing
    end
end

@inline function mpi_split(comm=mpi_comm(); color=0, key=0)
    comm_split = MPI.Comm_split(comm, color, key)
    MPI_INSTANCE_INITIALIZED[] = true
    MPI_COMM_INSTANCE[] = comm_split
    MPI_INSTANCE[] = color

    comm_shared = MPI.Comm_split(comm, MPI.Comm_rank(comm_split), 0)
    MPI_COMM_SHARED[] = comm_shared
    return comm_split
end

@inline function mpi_comm()
    mpi_init()
    return MPI.COMM_WORLD
end

@inline function mpi_comm_instance()
    mpi_init()
    return MPI_COMM_INSTANCE[]
end

@inline function mpi_comm_shared()
    mpi_init()
    return MPI_COMM_SHARED[]
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

@inline function mpi_recv!(args...; kwargs...)
    return MPI.Recv!(args...; kwargs...)
end

@inline function mpi_srecv(args...; kwargs...)
    return MPI.recv(args...; kwargs...)
end

@inline function mpi_irecv!(args...; kwargs...)
    return MPI.Irecv!(args...; kwargs...)
end

@inline function mpi_wait!(args...)
    return MPI.Wait!(args...)
end

@inline function mpi_waitall(args...)
    return MPI.Waitall(args...)
end

@inline function mpi_waitall!(args...)
    return MPI.Waitall!(args...)
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

@inline function mpi_datatype(::Type{T}) where {T}
    return MPI.Datatype(T)
end

@inline function mpi_buffer(args...)
    return MPI.Buffer(args...)
end

@inline function mpi_write_at(fp, offset, data)
    return MPI.File.write_at(fp, offset, data)
end

@inline mpi_make_transferrable(x) = mpi_make_transferrable(x, MPI_IS_GPUAWARE)
@inline mpi_make_transferrable(x::Array, ::Val{false}) = x # CPU
@inline mpi_make_transferrable(x::Array, ::Val{true}) = x
@inline mpi_make_transferrable(x, ::Val{false}) = Array(x) # GPU
@inline mpi_make_transferrable(x, ::Val{true}) = x

function instance_from_rank(mpi_rank::Int, numinstances::Int)
    @assert mod(mpi_size(), numinstances) == 0 "ranks not evenly assignable to instances"
    # Calculate how many ranks per instance
    ranks_per_instance = div(mpi_size(), numinstances)
    # Calculate which instance this rank belongs to
    return div(mpi_rank, ranks_per_instance)
end

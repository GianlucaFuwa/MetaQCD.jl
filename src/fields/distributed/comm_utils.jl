@inline distributed_reduce(var, ::Any, ::AbstractField) = var

@inline function distributed_reduce(var, op, u::AbstractMPIField)
    return mpi_allreduce(var, op, u.topology.comm_cart)
end

@inline function shrink_bulk_tup(bulk, stencil_radius)
    new_ranges = ntuple(Val(4)) do i
        irange = bulk.indices[i]
        new_first = first(irange) + stencil_radius[1]
        new_last = last(irange) - stencil_radius[1]
        @assert new_first <= new_last
        range(new_first, new_last)
    end

    return CartesianIndices(new_ranges)
end

@inline function shrink_bulk(bulk::CartesianIndices{4}, stencil_radius)
    new_ranges = ntuple(Val(4)) do i
        irange = bulk.indices[i]
        new_first = first(irange) + stencil_radius
        new_last = last(irange) - stencil_radius
        @assert new_first <= new_last
        range(new_first, new_last)
    end

    return CartesianIndices(new_ranges)
end

@inline function shrink_bulk(bulk::CartesianIndices{5}, stencil_radius)
    new_ranges = ntuple(Val(4)) do i
        irange = bulk.indices[i+1]
        new_first = first(irange) + stencil_radius
        new_last = last(irange) - stencil_radius
        @assert new_first <= new_last
        range(new_first, new_last)
    end
    
    first_range = bulk.indices[1]
    return CartesianIndices((first_range, new_ranges...))
end

function cooperative_wait(task::Task)
    while !Base.istaskdone(task)
        Utils.MPI.Iprobe(mpi_comm_instance())
        yield()
    end

    try
        wait(task)
    catch err
        error(err)
    end
    return nothing
end

function get_recv_task(::Type{backend}, recv_reqs::Vector{Utils.MPI.Request}) where {backend}
    return Base.Threads.@spawn begin
        KA.priority!(backend, :high)
        Base.wait.(recv_reqs)
        KA.synchronize(backend())
    end
end

function get_send_task(::Type{backend}, send_reqs::Vector{Utils.MPI.Request}) where {backend}
    return Base.Threads.@spawn begin
    end
end

function next_div32(n::Integer)
    return ((n ÷ 32) + 1) * 32
end

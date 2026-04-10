const HIDE_COMMS = Val(@load_preference("MPI_HIDE_COMMUNICATION", false))
const TUNE_KERNELS = Val(@load_preference("TUNE_KERNELS", false))

const KERNEL_CACHE::Dict{String,Int64} = Dict{String,Int64}() # function name => block size
const MAX_SHMEM = Base.RefValue{Int64}(0)

function groupreduce end    #
function threadidx end      #
function groupidx end       # These functions need to be overwritten in the extension
function groupdim end       # file of the GPU backends
function griddim end        #
function SharedMemory end   #

# From KernelAbstractions.jl
macro localmem(T, dims)
    id = gensym("static_shmem")
    return quote
        $SharedMemory($(esc(T)), Val($(esc(dims))), Val($(QuoteNode(id))))
    end
end

function parallelfor(
    f, # Kernel function (usually anonymous functions defined with "do" block)
    itr,
    ::Type{B}, # backend
    ::Val{M}, # whether field is mpi-distributed
    to_validate::Tuple,
    invalidated::Tuple,
    captured::Tuple;
    block_size=min(256, length(itr)),
    do_edges::Val{E}=Val(false),
    stream=default_stream(B())
) where {B,M,E}
    return parallelfor(
        f, itr, B, Val(M), HIDE_COMMS, to_validate, invalidated, captured;
        block_size, do_edges, stream
    )
end

function parallelfor(
    f,
    itr,
    ::Type{B},
    ::Val{M},
    ::Val{hide},
    to_validate::Tuple,
    invalidated::Tuple,
    captured::Tuple;
    block_size=min(256, length(itr)),
    do_edges::Val{E}=Val(false),
    stream=default_stream(B())
) where {B,M,hide,E}
        if M && (hide && B!=CPU) && length(to_validate) > 0
        # Drain the default (per-thread) stream before enqueuing the inner-bulk
        # kernel on the low-priority stream.  The *previous* parallelfor call
        # wrote border rows of the input field(s) on the PTDS (default stream);
        # those rows are read as stencil neighbours by the inner-bulk kernel.
        # Without this fence the two streams race: PTDS (normal priority) and
        # cu_streams[1] (low priority) have no implicit ordering on the GPU.
        synchronize(B(), default_stream(B()))
        # launch inner comp (async)
        hw, idx = findmin(get_halo_width, to_validate)
        inner_bulk = shrink_bulk(itr, hw, to_validate[1].topology.numprocs_cart)
        new_block_size = min(block_size, min(256, length(inner_bulk)))
        _parallelfor(f, captured, inner_bulk, B, new_block_size; stream=get_stream(B(), 1))

        # do exchange 
        start_halo_update!(to_validate; do_edges)

        # finish inner comp (to avoid resource contention)
        synchronize(B(), get_stream(B(), 1))

        # launch outer comp
        border_iterators = to_validate[idx].topology.border_iterators
        new_block_size = min(block_size, min(256, length(border_iterators[1])))
        _parallelfor(f, captured, border_iterators, B, new_block_size)
    elseif M && (!hide || B==CPU) && length(to_validate) > 0
        update_halo!(to_validate; do_edges)
        _parallelfor(f, captured, itr, B, block_size; stream)
    else
        _parallelfor(f, captured, itr, B, block_size; stream)
    end

    invalidate_halo!.(invalidated)
    return nothing
end

function _parallelfor(f, captured, itr, ::Type{CPU}, ::Int; kwargs...)
    @batch for i in eachindex(IndexLinear(), itr)
        @inbounds site = itr[i]
        @inline f(site, captured)
    end

    return nothing
end

function _parallelfor(f, captured, itrs::Tuple, ::Type{CPU}, ::Int; kwargs...)
    for itr in itrs
        @batch for i in eachindex(IndexLinear(), itr)
            @inbounds site = itr[i]
            @inline f(site, captured)
        end
    end

    return nothing
end

function _parallelfor(
    f, captured, itr, ::Type{backend}, block_size::Int=min(256, length(itr));
    stream=default_stream(backend())
) where {backend}
    @assert block_size > 0
    itr_tup = itr isa Tuple ? itr : (itr,)
    launch_foreachindex_global!(backend(), f, captured, itr_tup, block_size, stream)
    return nothing
end

"""
    launch_foreachindex_global!(
        backend, f, captured, itr::Tuple, groupsize, stream=default_stream(backend)
    )
Call the kernel function `f` with captured input arguments `captured` on `backend`
using the queue `stream` on each index of the iterators `itr`.
`itr` has to be a tuple of iterable objects that specify the indices or sites at which the
kernel is to be called.

The API calls for each backend are defined in their respective extension files under the 
directory ../ext/
"""
function launch_foreachindex_global! end

# KERNEL:
@inline function _foreachindex_global!(f, captured, itr)
    i = threadidx().x + (groupidx().x - Int32(1)) * groupdim().x

    if i <= length(itr)
        @inbounds site = itr[i]
        @inline f(site, captured)
    end

    return nothing
end

@inline function _foreachindex_global!(f, captured, itr...)
    ithread = threadidx().x + (groupidx().x - Int32(1)) * groupdim().x
    itr_lengths = length.(itr)

    if ithread <= sum(itr_lengths)
        iitr, i = get_iterator_index(ithread, itr_lengths)
        @inbounds site = itr[iitr][i]
        @inline f(site, captured)
    end

    return nothing
end

function parallelfor_sum(
    f,
    itr,
    init,
    ::Type{B},
    ::Val{M},
    to_validate::Tuple,
    invalidated::Tuple,
    captured::Tuple;
    block_size=min(256, length(itr)),
    do_edges::Val{E}=Val(false),
    stream=default_stream(B())
) where {B,M,E}
    return parallelfor_sum(
        f, itr, init, B, Val(M), HIDE_COMMS, to_validate, invalidated, captured;
        block_size, do_edges, stream
    )
end

function parallelfor_sum(
    f,
    itr,
    init,
    ::Type{B},
    ::Val{M},
    ::Val{hide},
    to_validate::Tuple,
    invalidated::Tuple,
    captured::Tuple;
    block_size=min(256, length(itr)),
    do_edges::Val{E}=Val(false),
    stream=default_stream(B())
) where {B,M,hide,E}
    if M && (hide && B!=CPU) && length(to_validate) > 0
        # inner work
        hw, idx = findmin(get_halo_width, to_validate)
        inner_bulk = shrink_bulk(itr, hw, to_validate[1].topology.numprocs_cart)
        new_block_size = min(block_size, min(256, length(inner_bulk)))
        result = _parallelfor_sum(f, captured, inner_bulk, init, B, new_block_size)#, stream=get_stream(B(), 1))

        start_halo_update!(to_validate; do_edges)

        # outer work
        border_iterators = to_validate[idx].topology.border_iterators
        new_block_size = min(block_size, min(256, length(border_iterators[1])))
        result += _parallelfor_sum(f, captured, border_iterators, init, B, new_block_size)
    elseif M && (!hide || B==CPU) && length(to_validate) > 0
        update_halo!(to_validate; do_edges)
        result = _parallelfor_sum(f, captured, itr, init, B, block_size; stream)
    else
        result = _parallelfor_sum(f, captured, itr, init, B, block_size; stream)
    end

    invalidate_halo!.(invalidated)
    return result
end

function _parallelfor_sum(f, captured, itr, init, ::Type{CPU}, ::Int; kwargs...)
    result = init

    @batch reduction = (+, result) for i in eachindex(IndexLinear(), itr)
        @inbounds site = itr[i]
        result += @inline f(init, site, captured)
    end

    return result
end

function _parallelfor_sum(f, captured, itrs::Tuple, init, ::Type{CPU}, ::Int; kwargs...)
    result = init

    for itr in itrs
        @batch reduction = (+, result) for i in eachindex(IndexLinear(), itr)
            @inbounds site = itr[i]
            result += @inline f(init, site, captured)
        end
    end

    return result
end

function _parallelfor_sum(
    f, captured, itr, init, ::Type{backend}, block_size::Int=min(256, length(itr));
    stream=default_stream(backend())
) where {backend}
    @assert block_size > 0
    itr_tup = itr isa Tuple ? itr : (itr,)
    result = launch_foreachindex_reduce_global!(
        backend(), init, +, f, captured, itr_tup, block_size, stream
    )
    return result
end

function launch_foreachindex_reduce_global! end

# KERNEL:
@inline function _foreachindex_reduce_global!(out, init, op, f, captured, itr)
    iblock = groupidx().x
    ithread = threadidx().x
    i = ithread + (iblock - Int32(1)) * groupdim().x

    if i <= length(itr)
        out_i = @inline f(init, itr[i], captured)
    else
        out_i = init
    end

    out_group = groupreduce(op, out_i, init)

    # We need the size of the grid here, in case we are launching the same kernel over
    # multiple iterators, since we still only use one out vector
    if ithread == 1
        @inbounds out[iblock] = op(out_group, out[iblock])
    end

    return nothing
end

function parallelfor_max(
    f, itr, init, ::Type{backend}, captured::Tuple, block_size::Int=min(256, length(itr))
) where {backend}
    if backend == CPU
        result = init

        @batch reduction = (max, result) for i in eachindex(IndexLinear(), itr)
            @inbounds site = itr[i]
            res = @inline f(init, site, captured)
            result = max(result, res)
        end

        return result
    else
        return launch_foreachindex_reduce_global!(
            backend(), init, max, f, captured, (itr,), block_size
        )
    end
end

@inline function get_iterator_index(ithread, itr_lengths)
    cumsums = (0, cumsum(itr_lengths)...)

    for i in 1:(length(itr_lengths))
        if ithread <= cumsums[i+1]
            iter_idx = i
            local_index = ithread - cumsums[i]
            return iter_idx, local_index
        end
    end

    throw(AssertionError("Out of Index in get_iterator_index"))
    return 0, 0
end

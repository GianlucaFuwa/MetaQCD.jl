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
    block_size=min(256, length(itr))
) where {B,M}
    return parallelfor(
        f, itr, B, Val(M), HIDE_COMMS, to_validate, invalidated, captured; block_size
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
    block_size=min(256, length(itr))
) where {B,M,hide}
    if M && hide && length(to_validate) > 0
        sendrecvtasks = start_halo_update!(to_validate; do_edges=Val(true))
        hw, idx = findmin(get_halo_width, to_validate)
        inner_bulk = shrink_bulk(itr, hw)
        new_block_size = min(block_size, min(256, length(inner_bulk)))

        # inner work
        _parallelfor(f, captured, inner_bulk, B, new_block_size)
        # wait for exchange to finish
        finalize_halo_update!(sendrecvtasks)
        # outer work
        border_iterators = to_validate[idx].topology.border_iterators
        new_block_size = min(block_size, min(256, length(border_iterators[1])))
        _parallelfor(f, captured, border_iterators, B, new_block_size)
    elseif M && !hide && length(to_validate) > 0
        update_halo!(to_validate)
        _parallelfor(f, captured, itr, B, block_size)
    else
        _parallelfor(f, captured, itr, B, block_size)
    end

    invalidate_halo!.(invalidated)
    return nothing
end

function _parallelfor(f, captured, itr, ::Type{CPU}, block_size)
    @batch for i in eachindex(IndexLinear(), itr)
        @inbounds site = itr[i]
        @inline f(site, captured)
    end

    return nothing
end

function _parallelfor(f, captured, itr, ::Type{backend}, block_size) where {backend}
    _foreachindex_gpu(f, captured, itr, backend(), block_size)
    return nothing
end

function _foreachindex_gpu(f, captured, itr, backend, block_size::Int=min(256, length(itr)))
    # name = nameof(f)
    # println(name)
    # GPU implementation
    @assert block_size > 0
    itr_tup = itr isa Tuple ? itr : (itr,)
    launch_foreachindex_global!(backend, f, captured, itr_tup, block_size)
    return nothing
end

function launch_foreachindex_global! end

# KERNEL:
@inline function _foreachindex_global!(f, captured, itr)
    i = threadidx().x + (groupidx().x - 0x1) * groupdim().x

    if i <= length(itr)
        @inbounds site = itr[i]
        @inline f(site, captured)
    end

    return nothing
end

@inline function _foreachindex_global!(
    f, captured, itr, block_dims,
    Gx::Int, Gy::Int, Gz::Int, Gt::Int
)
    nx, ny, nz, nt = ntuple(i -> length(itr.indices[i]), Val(4))
    Bx, By, Bz, _ = block_dims
    tx = threadidx().x
    ty = threadidx().y
    tz = threadidx().z

    pbx = groupidx().x
    pby = groupidx().y
    pbz = groupidx().z   # this is physical grid.z = Gz * Gt

    # --- recover logical block coords: block_z in [1..Gz], block_t in [1..Gt]
    # linear index (0-based) of the physical z-slab
    pz_lin0 = pbz - 1
    # block_z is pz_lin0 % Gz, block_t is pz_lin0 ÷ Gz
    block_z = Int(mod(pz_lin0, Gz)) + 1
    block_t = Int(pz_lin0 ÷ Gz) + 1

    # --- compute global coords (1-based) ---
    gx = (pbx - 1) * Bx + tx
    gy = (pby - 1) * By + ty
    gz = (block_z - 1) * Bz + tz
    gt = block_t

    # --- linear index in column-major (x fastest) ---
    # ensure coords in bounds before converting
    i = gx + (gy - 1) * nx + (gz - 1) * (nx * ny) + (gt - 1) * (nx * ny * nz)
    if i <= length(itr)
        @inbounds site = itr[i]
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
    block_size=min(256, length(itr))
) where {B,M}
    return parallelfor_sum(
        f, itr, init, B, Val(M), HIDE_COMMS, to_validate, invalidated, captured; block_size
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
    block_size=min(256, length(itr))
) where {B,M,hide}
    if M && hide && length(to_validate) > 0
        sendrecvtasks = start_halo_update!(to_validate; do_edges=Val(true))
        hw, idx = findmin(get_halo_width, to_validate)
        inner_bulk = shrink_bulk(itr, hw)
        new_block_size = min(block_size, min(256, length(inner_bulk)))

        # inner work
        result = _parallelfor_sum(f, captured, inner_bulk, init, B, new_block_size)
        # wait for exchange to finish
        finalize_halo_update!(sendrecvtasks)
        # outer work
        border_iterators = to_validate[idx].topology.border_iterators
        new_block_size = min(block_size, min(256, length(border_iterators[1])))
        result += _parallelfor_sum(f, captured, border_iterators, init, B, new_block_size)
    elseif M && !hide && length(to_validate) > 0
        update_halo!(to_validate)
        result = _parallelfor_sum(f, captured, itr, init, B, block_size)
    else
        result = _parallelfor_sum(f, captured, itr, init, B, block_size)
    end

    invalidate_halo!.(invalidated)
    return result
end

function _parallelfor_sum(f, captured, itr, init, ::Type{CPU}, block_size)
    result = init

    @batch reduction = (+, result) for i in eachindex(IndexLinear(), itr)
        @inbounds site = itr[i]
        result += @inline f(init, site, captured)
    end

    return result
end

function _parallelfor_sum(f, captured, itr, init, ::Type{backend}, block_size) where {backend}
    return _foreachindex_reduce_gpu(init, +, f, captured, itr, backend, block_size)
end

function _foreachindex_reduce_gpu(
    out, op, f, captured, itr, ::Type{backend}, block_size::Int=min(256, length(itr))
) where {backend}
    # GPU implementation
    @assert block_size > 0
    itr_tup = itr isa Tuple ? itr : (itr,)
    result = launch_foreachindex_reduce_global!(
        backend(), out, op, f, captured, itr_tup, block_size
    )
    return result
end

function launch_foreachindex_reduce_global! end

# KERNEL:
@inline function _foreachindex_reduce_global!(out, init, op, f, captured, itr, itr_idx)
    N = griddim().x
    iblock = groupidx().x
    ithread = threadidx().x
    i = ithread + (iblock - 0x1) * groupdim().x

    if i <= Int32(length(itr))
        out_i = @inline f(init, itr[i], captured)
    else
        out_i = init
    end

    out_group = groupreduce(op, out_i, init)

    # We need the size of the grid here, in case we are launching the same kernel over
    # multiple iterators, since we still only use one out vector
    if ithread == 1
        @inbounds out[iblock + N*(itr_idx-0x1)] = out_group
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
        return _foreachindex_reduce_gpu(init, max, f, captured, itr, backend, block_size)
    end
end

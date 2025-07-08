# This determines whether communication should be hidden in mpi halo exchange
# can be set in a LocalPreferences.toml file within the projects directory, like:
# [MetaQCD]
# MPI_HIDE_COMMUNICATION = true
const HIDE_COMMS = Val(@load_preference("MPI_HIDE_COMMUNICATION", false))

function parallelfor(
    f,
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
        sendrecvtasks = start_halo_update!(to_validate; do_edges=Val(false))
        hw, idx = findmin(get_halo_width, to_validate)
        inner_bulk = shrink_bulk(itr, hw)
        new_block_size = min(block_size, min(256, length(inner_bulk)))

        # inner work
        _parallelfor(f, captured, inner_bulk, B, new_block_size)
        # wait for exchange to finish
        finalize_halo_update!(sendrecvtasks)
        # outer work
        outer_bulk = to_validate[idx].topology.flat_border_sites
        _parallelfor(f, captured, outer_bulk, B, new_block_size)
    elseif M && !hide && length(to_validate) > 0
        update_halo!(to_validate)
        _parallelfor(f, captured, itr, B, block_size)
    else
        _parallelfor(f, captured, itr, B, block_size)
    end

    invalidate_halo!.(invalidated)
    return nothing
end

function _parallelfor(f, captured, itr, ::Type{backend}, block_size) where {backend}
    if backend == CPU
        @batch for i in eachindex(IndexLinear(), itr)
            @inbounds site = itr[i]
            @inline f(site, captured...)
        end
    else
        _foreachindex_gpu(f, itr, backend(), block_size)
    end

    return nothing
end

function _foreachindex_gpu(f, itr, backend::GPU, block_size::Int=min(256, length(itr)))
    # name = nameof(f)
    # println(name)
    # GPU implementation
    @assert block_size > 0
    blocks = (length(itr) + block_size - 1) ÷ block_size
    kernel = _foreachindex_global!(backend)
    kernel(f, itr; ndrange=length(itr))
    return nothing
end

@kernel inbounds=true unsafe_indices=true function _foreachindex_global!(
    f, itr
)
    # Calculate global index
    N = @groupsize()[1]
    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear)
    i = ithread + (iblock - 0x1) * N

    if i <= length(itr)
        f(itr[i])
    end
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
        sendrecvtasks = start_halo_update!(to_validate; do_edges=Val(false))
        hw, idx = findmin(get_halo_width, to_validate)
        inner_bulk = shrink_bulk(itr, hw)
        new_block_size = min(block_size, min(256, length(inner_bulk)))

        # inner work
        result = _parallelfor_sum(f, captured, inner_bulk, init, B, new_block_size)
        # wait for exchange to finish
        finalize_halo_update!(sendrecvtasks)
        # outer work
        outer_bulk = to_validate[idx].topology.flat_border_sites
        new_block_size = min(block_size, min(256, length(outer_bulk)))
        result += _parallelfor_sum(f, captured, outer_bulk, init, B, new_block_size)
    elseif M && !hide && length(to_validate) > 0
        update_halo!(to_validate)
        result = _parallelfor_sum(f, captured, itr, init, B, block_size)
    else
        result = _parallelfor_sum(f, captured, itr, init, B, block_size)
    end

    invalidate_halo!.(invalidated)
    return result
end

function _parallelfor_sum(f, captured, itr, init, ::Type{backend}, block_size) where {backend}
    if backend == CPU
        result = init

        @batch reduction = (+, result) for i in eachindex(IndexLinear(), itr)
            @inbounds site = itr[i]
            result += @inline f(init, site, captured...)
        end

        return result
    else
        return _foreachindex_reduce_gpu(init, +, f, itr, backend, block_size)
    end
end

function parallelfor_max(
    f, itr, init, ::Type{backend}, block_size::Int=min(256, length(itr))
) where {backend}
    if backend == CPU
        result = init

        @batch reduction = (max, result) for i in eachindex(IndexLinear(), itr)
            @inbounds site = itr[i]
            res = @inline f(init, site)
            result = max(result, res)
        end

        return result
    else
        return _foreachindex_reduce_gpu(init, max, f, itr, backend, block_size)
    end
end

function _foreachindex_reduce_gpu(
    out, op, f, itr, ::Type{backend}, block_size::Int=min(256, length(itr))
) where {backend}
    # name = nameof(f)
    # println(name)
    # GPU implementation
    @assert block_size > 0
    blocks = (length(itr) + block_size - 1) ÷ block_size
    out_vec = KA.zeros(backend(), typeof(out), blocks)
    kernel = _foreachindex_reduce_global!(backend(), block_size)
    kernel(out_vec, out, op, f, itr; ndrange=(block_size * blocks,))
    return reduce(op, out_vec)
end

@kernel inbounds=true unsafe_indices=true function _foreachindex_reduce_global!(
    out, init, op, f, itr
)
    # Calculate global index
    N = @groupsize()[1]
    iblock = @index(Group, Linear)
    ithread = @index(Local, Linear)
    i = ithread + (iblock - 0x1) * N

    if i <= length(itr)
        out_i = f(init, itr[i])
    end

    out_group = @groupreduce(op, out_i, init)

    ithread = @index(Local)
    if ithread == 1
        @inbounds out[iblock] = out_group
    end
end

simple_tune(itr, args...) = min(256, length(itr))

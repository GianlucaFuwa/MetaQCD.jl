function parallelfor(
    f, itr, ::Type{backend}, args...; block_size::Int=min(256, length(itr))
) where {backend}
    if backend == CPU
        @batch for i in eachindex(IndexLinear(), itr)
            @inbounds site = itr[i]
            @inline f(site)
        end
    else
        _foreachindex_gpu(f, itr, backend(); block_size)
    end

    return nothing
end

function _foreachindex_gpu(f, itr, backend::GPU; block_size::Int=min(256, length(itr)))
    # GPU implementation
    @assert block_size > 0
    blocks = (length(itr) + block_size - 1) ÷ block_size
    _foreachindex_global!(backend, block_size)(f, itr, ndrange=(block_size * blocks,))
    return nothing
end

@kernel inbounds=true cpu=false unsafe_indices=true function _foreachindex_global!(
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
    f, itr, init, ::Type{backend}, args...; block_size::Int=min(256, length(itr))
) where {backend}
    if backend == CPU
        result = init

        @batch reduction = (+, result) for i in eachindex(IndexLinear(), itr)
            @inbounds site = itr[i]
            result += @inline f(init, site)
        end

        return result
    else
        return _foreachindex_reduce_gpu(init, +, f, itr, backend; block_size)
    end
end

function _foreachindex_reduce_gpu(
    out, op, f, itr, ::Type{backend}; block_size::Int=min(256, length(itr))
) where {backend}
    # GPU implementation
    @assert block_size > 0
    blocks = (length(itr) + block_size - 1) ÷ block_size
    out_vec = KA.zeros(backend(), typeof(out), blocks)
    _foreachindex_reduce_global!(
        backend(), block_size)(out_vec, out, op, f, itr, ndrange=(block_size * blocks,)
    )
    return reduce(op, out_vec)
end

@kernel inbounds=true cpu=false unsafe_indices=true function _foreachindex_reduce_global!(
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
        out[iblock] = out_group
    end
end


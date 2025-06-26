"""
    @groupreduce(op, val, neutral, groupsize)
Copied from https://github.com/JuliaGPU/KernelAbstractions.jl/pull/421 \\
Reduce values across a block
- `op`: the operator of the reduction
- `val`: value that each thread contibutes to the values that need to be reduced
- `neutral`: value of the operator, so that `op(netural, neutral) = neutral``
- `groupsize` (optional): specify the groupsize. If not specified @groupsize is used but
this is generally slower.
"""
macro groupreduce(op, val, neutral)
    quote
        __groupreduce(
            $(esc(:__ctx__)),
            $(esc(op)),
            $(esc(val)),
            $(esc(neutral)),
            Val(prod($(KernelAbstractions.groupsize)($(esc(:__ctx__))))),
        )
    end
end

macro groupreduce(op, val, neutral, groupsize)
    quote
        __groupreduce(
            $(esc(:__ctx__)), $(esc(op)), $(esc(val)), $(esc(neutral)), $(esc(groupsize))
        )
    end
end

@inline function __groupreduce(__ctx__, op, val::T, neutral, ::Val{gsize}) where {T,gsize}
    idx_in_group = @index(Local)

    localmem = @localmem(T, gsize)

    @inbounds localmem[idx_in_group] = val

    # perform the reduction
    d = 1
    while d < gsize
        @synchronize()
        index = 2 * d * (idx_in_group - 1) + 1
        @inbounds if index <= gsize
            other_val = if index + d <= gsize
                localmem[index + d]
            else
                neutral
            end
            localmem[index] = op(localmem[index], other_val)
        end
        d *= 2
    end

    # load the final value on the first thread
    if idx_in_group == 1
        val = @inbounds localmem[idx_in_group]
    end

    return val
end


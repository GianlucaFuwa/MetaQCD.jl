"""
    @groupreduce(op, val, neutral, groupsize)
Copied from https://github.com/JuliaGPU/KernelAbstractions.jl/pull/421 \\
Reduce values across a block
- `op`: the operator of the reduction
- `val`: value that each thread contibutes to the values that need to be reduced
- `neutral`: value of the operator, so that `op(netural, neutral) = neutral``
- `groupsize` (optional): specify the groupszie. If not specified @groupsize is used but
this is generally slower.
"""
macro groupreduce(op, val, neutral)
    quote
        $__groupreduce($(esc(:__ctx__)),$(esc(op)), $(esc(val)), $(esc(neutral)),
                       Val(prod(KernelAbstractions.groupsize($(esc(:__ctx__))))))
    end
end

macro groupreduce(op, val, neutral, groupsize)
    quote
        $__groupreduce($(esc(:__ctx__)),$(esc(op)), $(esc(val)), $(esc(neutral)),
                       $(esc(groupsize)))
    end
end

@inline function __groupreduce(__ctx__, op, val::T, neutral, ::Val{GS}) where {T,GS}
    idx_in_group = @index(Local)

    localmem = @localmem(T, GS)

    @inbounds localmem[idx_in_group] = val

    # perform the reduction
    d = 1
    while d < GS
        @synchronize()
        index = 2 * d * (idx_in_group-1) + 1
        @inbounds if index <= GS
            other_val = if index + d <= GS
                localmem[index+d]
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

using MacroTools: @capture, prewalk, postwalk, splitdef, combinedef

# @inline function tile(NX::Integer, NY::Integer, NZ::Integer, NT::Integer)
#     maxn = 1024
#     num1 = min(maxn, NX)
#     num1 = ld(NX, num1)
#     maxn = fld(maxn, num1)
#     num2 = min(maxn, NY)
#     num2 = ld(NY, num2)
#     maxn = fld(maxn, num2)
#     num3 = min(maxn, NZ)
#     num3 = ld(NZ, num3)
#     maxn = fld(maxn, num3)
#     num4 = min(maxn, NT)
#     num4 = ld(NT, num4)
#     return num4, num3, num2, num1
# end

# function ld(x::Integer, y::Integer)
#     z = y
#     while z > 0
#         if x % z == 0
#             return z
#         end
#         z -= 1
#     end
#     return 1
# end

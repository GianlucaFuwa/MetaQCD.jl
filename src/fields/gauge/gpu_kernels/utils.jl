"""
	@latmap(kernel, U, args...)
Apply `kernel` on each element in `U` not following any specific pattern.
This assumes that no two iterations can influence each other.
"""
macro latmap(kernel, U, args...)
    quote
        $__latmap($(esc(kernel)), $(esc(U)), $(map(esc, args)...))
    end
end

@inline function __latmap(kernel, U::Abstractfield{GPUD}, args...)
    backend = get_backend(U)
    ndrange = size(U)[2:end]
    workgroupsize = (4, 4, 4, 4)
    compiled_kernel! = kernel(backend, workgroupsize)

    if length(args) == 0
        compiled_kernel!(U.U, ndrange=ndrange)
    else
        compiled_kernel!(U.U, get_raws(args...)..., ndrange=ndrange)
    end

    synchronize(backend)
    return nothing
end

"""
	@checker2map(kernel, U, args...)
Apply `kernel` on each element in `U` in a checkerboard pattern.
This assumes that the maximum stencil radius is 1.
"""
macro checker2map(kernel, U, args...)
    quote
        $__checker2map($(esc(kernel)), $(esc(U)), $(map(esc, args)...))
    end
end

@inline function __checker2map(kernel, U::Abstractfield{GPUD}, args...)
    backend = get_backend(U)
	NX, NY, NZ, NT = size(U)[2:end]
    @assert(mod.((NX, NY, NZ, NT), 2) == (0, 0, 0, 0),
        "CB2 only works for side lengths that are multiples of 2")
	ndrange = (NY, NZ, NT)
	workgroupsize = (4, 4, 4)

    compiled_kernel! = kernel(backend, workgroupsize)
    for μ in 1:4
        for pass in 1:2
            compiled_kernel!(U.U, μ, pass, get_raws(args...)..., ndrange=ndrange)
            synchronize(backend)
        end
    end
    return nothing
end

"""
	@checker4map(kernel, U, args...)
Apply `kernel` on each element in `U` splitting the lattice into 4 equal tiles.
This assumes that the maximum stencil radius is 2.
"""
macro checker4map(kernel, U, args...)
    quote
        $__checker4map($(esc(kernel)), $(esc(U)), $(map(esc, args)...))
    end
end

@inline function __checker4map(kernel, U::Abstractfield{GPUD}, args...)
    backend = get_backend(U)
	NX, NY, NZ, NT = size(U)[2:end]
    @assert(mod.((NX, NY, NZ, NT), 4) == (0, 0, 0, 0),
        "CB4 only works for side lengths that are multiples of 4")
    ndrange = (NY, NZ, NT)
	workgroupsize = (4, 4, 4)

    compiled_kernel! = kernel(backend, workgroupsize)
    for μ in 1:4
        for pass in 1:4
            compiled_kernel!(U.U, μ, pass, get_raws(args...)..., ndrange=ndrange)
            synchronize(backend)
        end
    end
    return nothing
end

"""
	@latreduce(op, kernel, U, args...)
Reduce `kernel` over `U` with the binary operator `op``not following any specific pattern.
This assumes that no two iterations can influence each other.
"""
macro latreduce(op, kernel, U, args...)
	quote
        $__latreduce($(esc(op)), $(esc(kernel)), $(esc(U)), $(map(esc, args)...))
    end
end

@inline function __latreduce(op, kernel, U::Abstractfield{GPUD,T}, args...) where {T}
    backend = get_backend(U)
    ndrange = size(U)[2:end]
    workgroupsize = (4, 4, 4, 4)
    numblocks = cld(U.NV, prod(workgroupsize))
    out = KA.zeros(backend, T, numblocks)
    compiled_kernel! = kernel(backend, workgroupsize)

    if length(args) == 0
        compiled_kernel!(out, U.U, zero(T), ndrange=ndrange)
    else
        compiled_kernel!(out, U.U, get_raws(args...)..., zero(T), ndrange=ndrange)
    end

    synchronize(backend)
    return reduce(op, out)
end

"""
	@checker2reduce(op, kernel, U, args...)
Reduce `kernel` over `U` with the binary operator `op` in a checkerboard pattern.
This assumes that the maximum stencil radius is 1.
"""
macro checker2reduce(op, kernel, U, args...)
    quote
        $__checker2reduce($(esc(op)), $(esc(kernel)), $(esc(U)), $(map(esc, args)...))
    end
end

@inline function __checker2reduce(op, kernel, U::Abstractfield{GPUD,T}, args...) where {T}
    backend = get_backend(U)
	NX, NY, NZ, NT = size(U)[2:end]
    @assert(mod.((NX, NY, NZ, NT), 2) == (0, 0, 0, 0),
        "CB2 only works for side lengths that are multiples of 2")
	ndrange = (NY, NZ, NT)
	workgroupsize = (4, 4, 4)
    numblocks = cld(U.NV, prod(workgroupsize))
    out = KA.zeros(backend, T, numblocks)

    compiled_kernel! = kernel(backend, workgroupsize)
    for μ in 1:4
        for pass in 1:2
            compiled_kernel!(out, U.U, μ, pass, get_raws(args...)..., zero(T),
                ndrange=ndrange)
            synchronize(backend)
        end
    end
    return reduce(op, out)
end

"""
	@checker4reduce(op, kernel, U, args...)
Reduce `kernel` over `U` with the binary operator `op` splitting the lattice into 4 equal
tiles. This assumes that the maximum stencil radius is 2.
"""
macro checker4reduce(op, kernel, U, args...)
    quote
        $__checker4reduce($(esc(op)), $(esc(kernel)), $(esc(U)), $(map(esc, args)...))
    end
end

@inline function __checker4reduce(op, kernel, U::Abstractfield{GPUD}, args...)
    backend = get_backend(U)
	NX, NY, NZ, NT = size(U)[2:end]
    @assert(mod.((NX, NY, NZ, NT), 4) == (0, 0, 0, 0),
        "CB4 only works for side lengths that are multiples of 4")
    ndrange = (NY, NZ, NT)
	workgroupsize = (4, 4, 4)
    numblocks = cld(U.NV, prod(workgroupsize))
    out = KA.zeros(backend, T, numblocks)

    compiled_kernel! = kernel(backend, workgroupsize)
    for μ in 1:4
        for pass in 1:4
            compiled_kernel!(out, U.U, μ, pass, get_raws(args...)..., zero(T),
                ndrange=ndrange)
            synchronize(backend)
        end
    end
    return reduce(op, out)
end

"""
    @groupreduce(op, val, neutral, use_subgroups)
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

@inline function get_raws(args...)
    fields = filter(x -> x isa Abstractfield, args)
    rest = filter(x -> !(x isa Abstractfield), args)
    return (ntuple(i -> fields[i].U, length(fields))..., rest...)
end

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

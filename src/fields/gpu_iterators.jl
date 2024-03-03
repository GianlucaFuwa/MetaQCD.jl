function __latmap(::Sequential, ::Val{COUNT}, f!::F, U::Abstractfield{B},
                  args...) where {COUNT,F,B<:GPU}
    COUNT==0 && return nothing
    # KernelAbstractions requires an ndrange (indices we iterate over) and a 
    # workgroupsize (number of threads in each workgroup / thread block on the GPU)
    ndrange = dims(U)
    workgroupsize = (4, 4, 4, 4) # 4^4 = 256 threads per workgroup should be fine
    kernel! = f!(B(), workgroupsize)
    # I couldn't be bothered working out how to make CUDA work with array wrappers such as
    # Abstractfield, so we extract the actual array from all Abstractfields in the args
    # In case some C/C++/Fortran programmer is reading this: We are not creating a copy
    # we are just getting the reference
    raw_args = get_raws(args...) 

    for _ in 1:COUNT
        kernel!(U.U, raw_args..., ndrange=ndrange)
        synchronize(B())
    end

    return nothing
end

function __latmap(::Checkerboard2, ::Val{COUNT}, f!::F, U::Abstractfield{B},
                  args...) where {COUNT,F,B<:GPU}
    COUNT==0 && return nothing
	NX, NY, NZ, NT = dims(U)
    @assert(mod.((NX, NY, NZ, NT), 2) == (0, 0, 0, 0),
        "CB2 only works for side lengths that are multiples of 2")
	ndrange = (NY, NZ, NT)
	workgroupsize = (4, 4, 4) # since we only have 64 threads per block, we can use heavier kernels
    kernel! = f!(B(), workgroupsize)
    raw_args = get_raws(args...)

    for _ in 1:COUNT
        for μ in 1:4
            for pass in 1:2
                kernel!(U.U, μ, pass, raw_args..., ndrange=ndrange)
                synchronize(B())
            end
        end
    end

    return nothing
end

function __latmap(::Checkerboard4, ::Val{COUNT}, f!::F, U::Abstractfield{B},
                  args...) where {COUNT,F,B<:GPU}
    COUNT==0 && return nothing
	NX, NY, NZ, NT = dims(U)
    @assert(mod.((NX, NY, NZ, NT), 4) == (0, 0, 0, 0),
        "CB4 only works for side lengths that are multiples of 4")
    ndrange = (NY, NZ, NT)
	workgroupsize = (4, 4, 4)
    kernel! = f!(B(), workgroupsize)
    raw_args = get_raws(args...)

    for _ in 1:COUNT
        for μ in 1:4
            for pass in 1:4
                kernel!(U.U, μ, pass, raw_args..., ndrange=ndrange)
                synchronize(B())
            end
        end
    end

    return nothing
end

function __latsum(::Sequential, ::Val{COUNT}, f!::F, U::Abstractfield{B},
                  args...) where {COUNT,F,B<:GPU}
    COUNT==0 && return 0.0
    ndrange = dims(U)
    workgroupsize = (4, 4, 4, 4)
    numblocks = cld(U.NV, prod(workgroupsize))
    out = KA.zeros(B(), Float64, numblocks)
    kernel! = f!(B(), workgroupsize)
    raw_args = get_raws(args...)

    for _ in 1:COUNT
        kernel!(out, U.U, raw_args..., ndrange=ndrange)
        synchronize(B())
    end

    return sum(out)
end

function __latsum(::Checkerboard2, ::Val{COUNT}, f!::F, U::Abstractfield{B},
                  args...) where {COUNT,F,B<:GPU}
    COUNT==0 && return 0.0
	NX, NY, NZ, NT = dims(U)
    @assert(mod.((NX, NY, NZ, NT), 2) == (0, 0, 0, 0),
        "CB2 only works for side lengths that are multiples of 2")
	ndrange = (NY, NZ, NT)
	workgroupsize = (4, 4, 4)
    numblocks = cld(U.NV, prod(workgroupsize))
    out = KA.zeros(B(), Float64, numblocks)
    kernel! = f!(B(), workgroupsize)
    raw_args = get_raws(args...)

    for _  in 1:COUNT
        for μ in 1:4
            for pass in 1:2
                kernel!(out, U.U, μ, pass, raw_args..., ndrange=ndrange)
                synchronize(B())
            end
        end
    end

    return sum(out)
end

function __latsum(::Checkerboard4, ::Val{COUNT}, f!::F, U::Abstractfield{B},
                  args...) where {COUNT,F,B<:GPU}
    COUNT==0 && return 0.0
	NX, NY, NZ, NT = dims(U)
    @assert(mod.((NX, NY, NZ, NT), 4) == (0, 0, 0, 0),
        "CB4 only works for side lengths that are multiples of 4")
    ndrange = (NY, NZ, NT)
	workgroupsize = (4, 4, 4)
    numblocks = cld(U.NV, prod(workgroupsize))
    out = KA.zeros(B(), Float64, numblocks)
    kernel! = f!(B(), workgroupsize)
    raw_args = get_raws(args...)

    for _ in 1:COUNT
        for μ in 1:4
            for pass in 1:4
                kernel!(out, U.U, μ, pass, raw_args..., ndrange=ndrange)
                synchronize(B())
            end
        end
    end

    return sum(out)
end

@inline function get_raws(args...)
    fields = filter(x -> x isa Abstractfield, args)
    rest = filter(x -> !(x isa Abstractfield), args)
    return (ntuple(i -> fields[i].U, length(fields))..., rest...)
end

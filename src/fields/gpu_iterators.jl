macro latmap(itr, C, f!, U, args...)
    quote
        $__latmap($(esc(itr)), $(esc(C)), $(esc(f!)), $(esc(U)), $(map(esc, args)...))
    end
end

function __latmap(
    ::Sequential, ::Val{COUNT}, f!::F, U::AbstractField{B}, args...
) where {COUNT,F,B<:GPU}
    COUNT == 0 && return nothing
    # KernelAbstractions requires an ndrange (indices we iterate over) and a 
    # workgroupsize (number of threads in each workgroup / thread block on the GPU)
    ndrange = local_dims(U)
    workgroupsize = (4, 4, 4, 4) # 4^4 = 256 threads per workgroup should be fine
    kernel! = f!(B(), workgroupsize)
    # I couldn't be bothered making GPUs work with array wrappers such as AbstractField
    # (see Adapt.jl), so we extract the actual array from all AbstractFields in the args
    raw_args = get_raws(args...)

    for _ in 1:COUNT
        kernel!(U.U, raw_args...; ndrange=ndrange)
        KA.synchronize(B())
    end

    return nothing
end

function __latmap(
    ::Sequential, ::Val{COUNT}, f!::F, ϕ_eo::SpinorfieldEO{B}, args...
) where {COUNT,F,B<:GPU}
    COUNT == 0 && return nothing
    # KernelAbstractions requires an ndrange (indices we iterate over) and a 
    # workgroupsize (number of threads in each workgroup / thread block on the GPU)
    NX, NY, NZ, NT = local_dims(ϕ_eo)
    @assert iseven(NT) "NT must be even for even-odd preconditioned fermions"
    ndrange = (NX, NY, NZ, div(NT, 2))
    workgroupsize = (4, 4, 4, 2)
    kernel! = f!(B(), workgroupsize)
    raw_args = get_raws(args...)

    for _ in 1:COUNT
        kernel!(ϕ_eo.parent.U, raw_args...; ndrange=ndrange)
        KA.synchronize(B())
    end

    return nothing
end

function __latmap(
    ::Checkerboard2, ::Val{COUNT}, f!::F, U::AbstractField{B}, args...
) where {COUNT,F,B<:GPU}
    COUNT == 0 && return nothing
    NX, NY, NZ, NT = local_dims(U)
    @assert(
        mod.((NX, NY, NZ, NT), 2) == (0, 0, 0, 0),
        "CB2 only works for side lengths that are multiples of 2"
    )
    ndrange = (NY, NZ, NT)
    workgroupsize = (4, 4, 4) # since we only have 64 threads per block, we can use heavier kernels
    kernel! = f!(B(), workgroupsize)
    raw_args = get_raws(args...)

    for _ in 1:COUNT
        for μ in 1:4
            for pass in 1:2
                kernel!(U.U, μ, pass, raw_args...; ndrange=ndrange)
                KA.synchronize(B())
            end
        end
    end

    return nothing
end

function __latmap(
    ::Checkerboard4, ::Val{COUNT}, f!::F, U::AbstractField{B}, args...
) where {COUNT,F,B<:GPU}
    COUNT == 0 && return nothing
    NX, NY, NZ, NT = local_dims(U)
    @assert(
        mod.((NX, NY, NZ, NT), 4) == (0, 0, 0, 0),
        "CB4 only works for side lengths that are multiples of 4"
    )
    ndrange = (NY, NZ, NT)
    workgroupsize = (4, 4, 4)
    kernel! = f!(B(), workgroupsize)
    raw_args = get_raws(args...)

    for _ in 1:COUNT
        for μ in 1:4
            for pass in 1:4
                kernel!(U.U, μ, pass, raw_args...; ndrange=ndrange)
                KA.synchronize(B())
            end
        end
    end

    return nothing
end

macro latsum(itr, C, f!, U, args...)
    quote
        $__latsum($(esc(itr)), $(esc(C)), $(esc(f!)), $(esc(U)), $(map(esc, args)...))
    end
end

function __latsum(
    ::Sequential, ::Val{COUNT}, ::Type{OutType}, f!::F, U::AbstractField{B}, args...
) where {COUNT,OutType,F,B<:GPU}
    COUNT == 0 && return 0.0
    ndrange = local_dims(U)
    workgroupsize = (4, 4, 4, 4)
    numblocks = cld(U.NV, prod(workgroupsize))
    out = KA.zeros(B(), OutType, numblocks)
    kernel! = f!(B(), workgroupsize)
    raw_args = get_raws(args...)

    for _ in 1:COUNT
        kernel!(out, U.U, raw_args...; ndrange=ndrange)
        KA.synchronize(B())
    end

    return sum(out)
end

function __latsum(
    ::Sequential, ::Val{COUNT}, ::Type{OutType}, f!::F, ϕ_eo::SpinorfieldEO{B}, args...
) where {COUNT,OutType,F,B<:GPU}
    COUNT == 0 && return 0.0
    NX, NY, NZ, NT = local_dims(ϕ_eo)
    @assert iseven(NT) "NT must be even for even-odd preconditioned fermions"
    ndrange = (NX, NY, NZ, div(NT, 2))
    workgroupsize = (4, 4, 4, 2)
    numblocks = cld(div(ϕ_eo.parent.NV, 2), prod(workgroupsize))
    out = KA.zeros(B(), OutType, numblocks)
    kernel! = f!(B(), workgroupsize)
    raw_args = get_raws(args...)

    for _ in 1:COUNT
        kernel!(out, ϕ_eo.parent.U, raw_args...; ndrange=ndrange)
        KA.synchronize(B())
    end

    return sum(out)
end

function __latsum(
    ::Checkerboard2, ::Val{COUNT}, f!::F, U::AbstractField{B}, args...
) where {COUNT,F,B<:GPU}
    COUNT == 0 && return 0.0
    NX, NY, NZ, NT = local_dims(U)
    @assert(
        mod.((NX, NY, NZ, NT), 2) == (0, 0, 0, 0),
        "CB2 only works for side lengths that are multiples of 2"
    )
    ndrange = (NY, NZ, NT)
    workgroupsize = (4, 4, 4)
    numblocks = cld(U.NV, prod(workgroupsize))
    out = KA.zeros(B(), Float64, numblocks)
    kernel! = f!(B(), workgroupsize)
    raw_args = get_raws(args...)
    numpasses = U isa SpinorfieldEO ? 2 : 1

    for _ in 1:COUNT
        for μ in 1:4
            for pass in 1:numpasses
                kernel!(out, U.U, μ, pass, raw_args...; ndrange=ndrange)
                KA.synchronize(B())
            end
        end
    end

    return sum(out)
end

function __latsum(
    ::Checkerboard4, ::Val{COUNT}, f!::F, U::AbstractField{B}, args...
) where {COUNT,F,B<:GPU}
    COUNT == 0 && return 0.0
    NX, NY, NZ, NT = local_dims(U)
    @assert(
        mod.((NX, NY, NZ, NT), 4) == (0, 0, 0, 0),
        "CB4 only works for side lengths that are multiples of 4"
    )
    ndrange = (NY, NZ, NT)
    workgroupsize = (4, 4, 4)
    numblocks = cld(U.NV, prod(workgroupsize))
    out = KA.zeros(B(), Float64, numblocks)
    kernel! = f!(B(), workgroupsize)
    raw_args = get_raws(args...)

    for _ in 1:COUNT
        for μ in 1:4
            for pass in 1:4
                kernel!(out, U.U, μ, pass, raw_args...; ndrange=ndrange)
                KA.synchronize(B())
            end
        end
    end

    return sum(out)
end

@inline function get_raws(args...)
    fields = filter(x -> x isa AbstractField, args)
    rest = filter(x -> !(x isa AbstractField), args)
    raw_fields = ntuple(
        i -> fields[i] isa SpinorfieldEO ? fields[i].parent.U : fields[i].U, length(fields)
    )
    return (raw_fields..., rest...)
end

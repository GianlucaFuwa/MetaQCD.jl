@inline function __latmap(::Sequential, ::Val{count}, f!::F, U::Abstractfield{GPUD},
    args...) where {F,count}
    count==0 && return nothing
    backend = get_backend(U)
    ndrange = size(U)[2:end]
    workgroupsize = (4, 4, 4, 4)
    kernel! = f!(backend, workgroupsize)

    for _ in 1:count
        kernel!(U.U, get_raws(args...)..., ndrange=ndrange)
        synchronize(backend)
    end

    return nothing
end

@inline function __latmap(::Checkerboard2, ::Val{count}, f!::F, U::Abstractfield{GPUD},
    args...) where {F,count}
    count==0 && return nothing
    backend = get_backend(U)
	NX, NY, NZ, NT = size(U)[2:end]
    @assert(mod.((NX, NY, NZ, NT), 2) == (0, 0, 0, 0),
        "CB2 only works for side lengths that are multiples of 2")
	ndrange = (NY, NZ, NT)
	workgroupsize = (4, 4, 4)
    kernel! = f!(backend, workgroupsize)

    for _ in 1:count
        for μ in 1:4
            for pass in 1:2
                kernel!(U.U, μ, pass, get_raws(args...)..., ndrange=ndrange)
                synchronize(backend)
            end
        end
    end

    return nothing
end

@inline function __latmap(::Checkerboard4, ::Val{count}, f!::F, U::Abstractfield{GPUD},
    args...) where {F,count}
    count==0 && return nothing
    backend = get_backend(U)
	NX, NY, NZ, NT = size(U)[2:end]
    @assert(mod.((NX, NY, NZ, NT), 4) == (0, 0, 0, 0),
        "CB4 only works for side lengths that are multiples of 4")
    ndrange = (NY, NZ, NT)
	workgroupsize = (4, 4, 4)
    kernel! = f!(backend, workgroupsize)

    for _ in 1:count
        for μ in 1:4
            for pass in 1:4
                kernel!(U.U, μ, pass, get_raws(args...)..., ndrange=ndrange)
                synchronize(backend)
            end
        end
    end

    return nothing
end

@inline function __latsum(::Sequential, ::Val{count}, f!::F, U::Abstractfield{GPUD,T},
    args...) where {count,F,T}
    count==0 && return zero(T)
    backend = get_backend(U)
    ndrange = size(U)[2:end]
    workgroupsize = (4, 4, 4, 4)
    numblocks = cld(U.NV, prod(workgroupsize))
    out = KA.zeros(backend, T, numblocks)
    kernel! = f!(backend, workgroupsize)

    for _ in 1:count
        kernel!(out, U.U, get_raws(args...)..., ndrange=ndrange)
        synchronize(backend)
    end

    return sum(out)
end

@inline function __latsum(::Checkerboard2, ::Val{count}, f!::F, U::Abstractfield{GPUD,T},
    args...) where {count,F,T}
    count==0 && return zero(T)
    backend = get_backend(U)
	NX, NY, NZ, NT = size(U)[2:end]
    @assert(mod.((NX, NY, NZ, NT), 2) == (0, 0, 0, 0),
        "CB2 only works for side lengths that are multiples of 2")
	ndrange = (NY, NZ, NT)
	workgroupsize = (4, 4, 4)
    numblocks = cld(U.NV, prod(workgroupsize))
    out = KA.zeros(backend, T, numblocks)
    kernel! = f!(backend, workgroupsize)

    for _  in 1:count
        for μ in 1:4
            for pass in 1:2
                kernel!(out, U.U, μ, pass, get_raws(args...)..., ndrange=ndrange)
                synchronize(backend)
            end
        end
    end

    return sum(out)
end

@inline function __latsum(::Checkerboard4, ::Val{count}, f!::F, U::Abstractfield{GPUD,T},
    args...) where {count,F,T}
    count==0 && return zero(T)
    backend = get_backend(U)
	NX, NY, NZ, NT = size(U)[2:end]
    @assert(mod.((NX, NY, NZ, NT), 4) == (0, 0, 0, 0),
        "CB4 only works for side lengths that are multiples of 4")
    ndrange = (NY, NZ, NT)
	workgroupsize = (4, 4, 4)
    numblocks = cld(U.NV, prod(workgroupsize))
    out = KA.zeros(backend, T, numblocks)
    kernel! = f!(backend, workgroupsize)

    for _ in 1:count
        for μ in 1:4
            for pass in 1:4
                kernel!(out, U.U, μ, pass, get_raws(args...)..., ndrange=ndrange)
                synchronize(backend)
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

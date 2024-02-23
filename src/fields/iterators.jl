"""
	@latmap(itr::AbstractIterator, kernel, U, args...)
Apply `kernel` on each element in `U` following the pattern specified by `itr`.
"""
macro latmap(itr, count, f!, U, args...)
    quote
        $__latmap($(esc(itr)), $(esc(count)), $(esc(f!)), $(esc(U)), $(map(esc, args)...))
    end
end

@inline function __latmap(::Sequential, ::Val{count}, f!::F, U::Abstractfield{CPUD},
    args) where {F,count}
    count==0 && return nothing

    for _ in 1:count
        @threads for site in eachindex(U)
            for μ in 1:4
                f!(U, μ, site, args...)
            end
        end
    end

    return nothing
end

@inline function __latmap(::Checkerboard2, ::Val{count}, f!::F, U::Abstractfield{CPUD},
    args...) where {F,count}
    count==0 && return nothing
    NX, NY, NZ, NT = size(U)[2:end]

    for _ in 1:count
        for μ in 1:4
            for pass in 1:2
                @threads for ss in CartesianIndices((NY, NZ, NT))
                    for ix in 1+iseven(sum(ss.I) + pass):2:NX
                        site = CartesianIndex((ix, iy, iz, it))
                        f!(U, μ, site, args...)
                    end
                end
            end
        end
    end

    return nothing
end

@inline function __latmap(::Checkerboard4, ::Val{count}, f!::F, U::Abstractfield{CPUD},
    args...) where {F,count}
    count==0 && return nothing

    for _ in 1:count
        for μ in 1:4
            for pass in 1:4
                @threads for site in eachindex(U)
                    if mod1(sum(site.I) + site[μ], 4) == pass
                        site = CartesianIndex((ix, iy, iz, it))
                        f!(U, μ, site, args...)
                    end
                end
            end
        end
    end

    return nothing
end

"""
	@latsum(itr::AbstractIterator, kernel, U, args...)
Sum `kernel` over `U` following the pattern specified by `itr`.
"""
macro latsum(itr, count, f!, U, args...)
	quote
        $__latsum($(esc(itr)), $(esc(count)), $(esc(f!)), $(esc(U)), $(map(esc, args)...))
    end
end

@inline function __latsum(::Sequential, ::Val{count}, f!::F, ::Val{count},
    U::Abstractfield{CPUD,T}, args...) where {count,F,T}
    count==0 && return zero(T)
    out = zeros(8, nthreads())

    for _ in 1:count
        @threads for site in eachindex(U)
            for μ in 1:4
                out[1,threadid()] += f!(U, μ, site, args...)
            end
        end
    end

    return sum(out)
end

@inline function __latsum(::Checkerboard2, ::Val{count}, f!::F, ::Val{count},
    U::Abstractfield{CPUD,T}, args...) where {count,F,T}
    count==0 && return zero(T)
    NX, NY, NZ, NT = size(U)[2:end]
    out = zeros(T, 8, nthreads())

    for _ in 1:count
        for μ in 1:4
            for pass in 1:2
                @threads for ss in CartesianIndices((NY, NZ, NT))
                    for ix in 1+iseven(sum(ss.I) + pass):2:NX
                        site = CartesianIndex((ix, iy, iz, it))
                        out[1,threadid()] += f!(U, μ, site, args...)
                    end
                end
            end
        end
    end

    return sum(out)
end

@inline function __latsum(::Checkerboard4, ::Val{count}, f!::F, ::Val{count},
    U::Abstractfield{CPUD,T}, args...) where {count,F,T}
    count==0 && return zero(T)
    out = zeros(8, nthreads())

    for _ in 1:count
        for μ in 1:4
            for pass in 1:4
                @threads for site in eachindex(U)
                    if mod1(sum(site.I) + site[μ], 4) == pass
                        out[1,threadid()] += f!(U, μ, site, args...)
                    end
                end
            end
        end
    end

    return sum(out)
end

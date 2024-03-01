"""
	@latmap(itr::AbstractIterator, kernel, U, args...)
Apply `kernel` on each element in `U` following the pattern specified by `itr`.
"""
macro latmap(itr, count, f!, U, args...)
    quote
        $__latmap($(esc(itr)), $(esc(count)), $(esc(f!)), $(esc(U)), $(map(esc, args)...))
    end
end

function __latmap(::Sequential, ::Val{COUNT}, f!::F, U::Abstractfield{CPU},
                  args) where {F,COUNT}
    COUNT==0 && return nothing

    for _ in 1:COUNT
        @batch for site in eachindex(U)
            for μ in 1:4
                f!(U, μ, site, args...)
            end
        end
    end

    return nothing
end

function __latmap(::Checkerboard2, ::Val{COUNT}, f!::F, U::Abstractfield{CPU},
                  args...) where {F,COUNT}
    COUNT==0 && return nothing
    NX, NY, NZ, NT = dims(U)

    for _ in 1:COUNT
        for μ in 1:4
            for pass in 1:2
                @batch for ss in CartesianIndices((NY, NZ, NT))
                    for ix in 1+iseven(sum(ss.I) + pass):2:NX
                        site = CartesianIndex((ix, ss.I...))
                        f!(U, μ, site, args...)
                    end
                end
            end
        end
    end

    return nothing
end

function __latmap(::Checkerboard4, ::Val{COUNT}, f!::F, U::Abstractfield{CPU},
                  args...) where {F,COUNT}
    COUNT==0 && return nothing

    for _ in 1:COUNT
        for μ in 1:4
            for pass in 1:4
                @batch for site in eachindex(U)
                    if mod1(sum(site.I) + site[μ], 4) == pass
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
macro latsum(itr, COUNT, f!, U, args...)
	quote
        $__latsum($(esc(itr)), $(esc(COUNT)), $(esc(f!)), $(esc(U)), $(map(esc, args)...))
    end
end

function __latsum(::Sequential, ::Val{COUNT}, f!::F, U::Abstractfield{CPU,T},
                  args...) where {COUNT,F,T}
    COUNT==0 && return 0.0
    out = 0.0

    for _ in 1:COUNT
        @batch reduction=(+, out) for site in eachindex(U)
            for μ in 1:4
                out += f!(U, μ, site, args...)
            end
        end
    end

    return out
end

function __latsum(::Checkerboard2, ::Val{COUNT}, f!::F, U::Abstractfield{CPU,T},
                  args...) where {COUNT,F,T}
    COUNT==0 && return 0.0
    NX, NY, NZ, NT = dims(U)
    out = 0.0

    for _ in 1:COUNT
        for μ in 1:4
            for pass in 1:2
                @batch reduction=(+, out) for ss in CartesianIndices((NY, NZ, NT))
                    for ix in 1+iseven(sum(ss.I) + pass):2:NX
                        site = CartesianIndex((ix, ss.I...))
                        out += f!(U, μ, site, args...)
                    end
                end
            end
        end
    end

    return out
end

function __latsum(::Checkerboard4, ::Val{COUNT}, f!::F, U::Abstractfield{CPU,T},
                  args...) where {COUNT,F,T}
    COUNT==0 && return 0.0
    out = 0.0

    for _ in 1:COUNT
        for μ in 1:4
            for pass in 1:4
                @batch reduction=(+, out) for site in eachindex(U)
                    if mod1(sum(site.I) + site[μ], 4) == pass
                        out += f!(U, μ, site, args...)
                    end
                end
            end
        end
    end

    return out
end

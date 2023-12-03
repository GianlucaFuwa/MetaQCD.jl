
abstract type AbstractIterator end

struct Sequential <: AbstractIterator end
struct SequentialMT <: AbstractIterator end
struct Checkerboard2 <: AbstractIterator end
struct Checkerboard2MT <: AbstractIterator end
struct Checkerboard4 <: AbstractIterator end
struct Checkerboard4MT <: AbstractIterator end

const threadlocals = zeros(Float64, 8nthreads())

function sweep!(::Sequential, ::Val{count}, f!::F, U, args...) where {F, count}
    for _ in 1:count
        for site in eachindex(U)
            for μ in 1:4
                f!(U, μ, site, args...)
            end
        end
    end

    return nothing
end

function sweep!(::SequentialMT, ::Val{count}, f!::F, U, args...) where {F, count}
    for _ in 1:count
        # need to use @threads instead of @batch, because @batch does not produce consistent
        # results on asynchronous tasks apparently
        @threads for site in eachindex(U)
            for μ in 1:4
                f!(U, μ, site, args...)
            end
        end
    end

    return nothing
end

function sweep!(::Checkerboard2, ::Val{count}, f!::F, U, args...) where {F, count}
    NX, NY, NZ, NT = size(U)

    for _ in 1:count
        for μ in 1:4
            for pass in 1:2
                for it in 1:NT
                    for iz in 1:NZ
                        for iy in 1:NY
                            for ix in 1+iseven(it + iz + iy + pass):2:NX
                                site = SiteCoords(ix, iy, iz, it)
                                f!(U, μ, site, args...)
                            end
                        end
                    end
                end
            end
        end
    end

    return nothing
end

function sweep!(::Checkerboard2MT, ::Val{count}, f!::F, U, args...) where {F, count}
    NX, NY, NZ, NT = size(U)

    for _ in 1:count
        for μ in 1:4
            for pass in 1:2
                # need to use @threads instead of @batch, because @batch does not produce
                # consistent results on asynchronous tasks apparently
                @threads for it in 1:NT
                    for iz in 1:NZ
                        for iy in 1:NY
                            for ix in 1+iseven(it + iz + iy + pass):2:NX
                                site = SiteCoords(ix, iy, iz, it)
                                f!(U, μ, site, args...)
                            end
                        end
                    end
                end
            end
        end
    end

    return nothing
end

function sweep!(::Checkerboard4, ::Val{count}, f!::F, U, args...) where {F, count}
    NX, NY, NZ, NT = size(U)
    numhits = 0
    hitlist = NTuple{5, Int64}[]

    for _ in 1:count
        for μ in 1:4
            for pass in 1:4
                for it in 1:NT
                    for iz in 1:NZ
                        for iy in 1:NY
                            # instead of checking if we are on an even site, we split the
                            # lattice into 4 sublattices that we update one after the other
                            for ix in 1:NX
                                site = SiteCoords(ix, iy, iz, it)
                                if mod1(it+iz+iy+ix + site[μ], 4)!=pass
                                    continue
                                end
                                numhits += 1
                                s = (μ, ix, iy, iz, it)
                                s∉hitlist && push!(hitlist, s)
                                f!(U, μ, site, args...)
                            end
                        end
                    end
                end
            end
        end
    end

    @assert (length(hitlist)==4U.NV && numhits==4U.NV) "We only hit $(length(hitlist)) / $(4U.NV) links"
    return nothing
end

function sweep!(::Checkerboard4MT, ::Val{count}, f!::F, U, args...)::Nothing where {F, count}
    NX, NY, NZ, NT = size(U)

    for _ in 1:count
        for μ in 1:4
            for pass in 1:4
                # need to use @threads instead of @batch, because @batch does not produce
                # consistent results on asynchronous tasks apparently
                @threads for it in 1:NT
                    for iz in 1:NZ
                        for iy in 1:NY
                            # instead of checking if we are on an even site, we split the
                            # lattice into 4 sublattices that we update one after the other
                            for ix in 1:NX
                                site = SiteCoords(ix, iy, iz, it)
                                if mod1(it+iz+iy+ix + site[μ], 4)!=pass
                                    continue
                                end
                                f!(U, μ, site, args...)
                            end
                        end
                    end
                end
            end
        end
    end

    return nothing
end

function sweep_reduce!(::Sequential, ::Val{count}, f!::F, U, args...) where {F, count}
    out = 0.0

    for _ in 1:count
        for site in eachindex(U)
            for μ in 1:4
                out += f!(U, μ, site, args...)
            end
        end
    end

    return out
end

function sweep_reduce!(::SequentialMT, ::Val{count}, f!::F, U, args...) where {F, count}
    count > 0 || return 0.0
    out = zeros(Float64, 8nthreads())

    for _ in 1:count
        @batch per=thread for site in eachindex(U)
            for μ in 1:4
                out[8threadid()] += f!(U, μ, site, args...)
            end
        end
    end

    return sum(out)
end

function sweep_reduce!(::Checkerboard2, ::Val{count}, f!::F, U, args...) where {F, count}
    NX, NY, NZ, NT = size(U)
    out = 0.0

    for _ in 1:count
        for μ in 1:4
            for pass in 1:2
                for it in 1:NT
                    for iz in 1:NZ
                        for iy in 1:NY
                            for ix in 1+iseven(it + iz + iy + pass):2:NX
                                site = SiteCoords(ix, iy, iz, it)
                                out += f!(U, μ, site, args...)
                            end
                        end
                    end
                end
            end
        end
    end

    return out
end

function sweep_reduce!(::Checkerboard2MT, ::Val{count}, f!::F, U, args...) where {F, count}
    count > 0 || return 0.0
    NX, NY, NZ, NT = size(U)
    out = zeros(Float64, 8nthreads())

    for _ in 1:count
        for μ in 1:4
            for pass in 1:2
                @batch per=thread for it in 1:NT
                    for iz in 1:NZ
                        for iy in 1:NY
                            for ix in 1+iseven(it + iz + iy + pass):2:NX
                                site = SiteCoords(ix, iy, iz, it)
                                out[8threadid()] += f!(U, μ, site, args...)
                            end
                        end
                    end
                end
            end
        end
    end

    return sum(out)
end

function sweep_reduce!(::Checkerboard4, ::Val{count}, f!::F, U, args...) where {F, count}
    NX, NY, NZ, NT = size(U)
    out = 0.0

    for _ in 1:count
        for μ in 1:4
            for pass in 1:4
                for it in 1:NT
                    for iz in 1:NZ
                        for iy in 1:NY
                            # instead of checking if we are on an even site, we split the
                            # lattice into 4 sublattices that we update one after the other
                            for ix in 1:NX
                                site = SiteCoords(ix, iy, iz, it)
                                if mod1(it+iz+iy+ix + site[μ], 4)!=pass
                                    continue
                                end
                                out += f!(U, μ, site, args...)
                            end
                        end
                    end
                end
            end
        end
    end

    return out
end

function sweep_reduce!(::Checkerboard4MT, ::Val{count}, f!::F, U, args...) where {F, count}
    count > 0 || return 0.0
    NX, NY, NZ, NT = size(U)
    out = zeros(Float64, 8nthreads())

    for _ in 1:count
        for μ in 1:4
            for pass in 1:4
                @batch per=thread for it in 1:NT
                    for iz in 1:NZ
                        for iy in 1:NY
                            # instead of checking if we are on an even site, we split the
                            # lattice into 4 sublattices that we update one after the other
                            for ix in 1:NX
                                site = SiteCoords(ix, iy, iz, it)
                                if mod1(it+iz+iy+ix + site[μ], 4)!=pass
                                    continue
                                end
                                out[8threadid()] += f!(U, μ, site, args...)
                            end
                        end
                    end
                end
            end
        end
    end

    return sum(out)
end

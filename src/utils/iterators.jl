
abstract type AbstractIterator end

struct Sequential <: AbstractIterator end
struct SequentialMT <: AbstractIterator end
struct Checkerboard2 <: AbstractIterator end
struct Checkerboard2MT <: AbstractIterator end
struct Checkerboard4 <: AbstractIterator end
struct Checkerboard4MT <: AbstractIterator end

function sweep!(::Sequential, count, f!, U, args...)
    for _ in 1:count
        for site in eachindex(U)
            for μ in 1:4
                f!(U, μ, site, args...)
            end
        end
    end

    return nothing
end

function sweep!(::SequentialMT, count, f!, U, args...)
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

function sweep!(::Checkerboard2, count, f!, U, args...)
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

function sweep!(::Checkerboard2MT, count, f!, U, args...)
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

function sweep!(::Checkerboard4, count, f!, U, args...)
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

function sweep!(::Checkerboard4MT, count, f!, U, args...)
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

function sweep_reduce!(::Sequential, count, f!, U, args...)
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

function sweep_reduce!(::SequentialMT, count, f!, U, args...)
    out = 0.0

    for _ in 1:count
        @batch per=thread threadlocal=0.0::Float64 for site in eachindex(U)
            for μ in 1:4
                threadlocal += f!(U, μ, site, args...)
            end
        end
        out += sum(threadlocal)
    end

    return sum(out)
end

function sweep_reduce!(::Checkerboard2, count, f!, U, args...)
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

function sweep_reduce!(::Checkerboard2MT, count, f!, U, args...)
    NX, NY, NZ, NT = size(U)
    out = 0.0

    for _ in 1:count
        for μ in 1:4
            for pass in 1:2
                @batch per=thread threadlocal=0.0::Float64 for it in 1:NT
                    for iz in 1:NZ
                        for iy in 1:NY
                            for ix in 1+iseven(it + iz + iy + pass):2:NX
                                site = SiteCoords(ix, iy, iz, it)
                                threadlocal += f!(U, μ, site, args...)
                            end
                        end
                    end
                end
                out += sum(threadlocal)
            end
        end
    end

    return out
end

function sweep_reduce!(::Checkerboard4, count, f!, U, args...)
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

function sweep_reduce!(::Checkerboard4MT, count, f!, U, args...)
    NX, NY, NZ, NT = size(U)
    out = 0.0

    for _ in 1:count
        for μ in 1:4
            for pass in 1:4
                @batch per=thread threadlocal=0.0::Float64 for it in 1:NT
                    for iz in 1:NZ
                        for iy in 1:NY
                            # instead of checking if we are on an even site, we split the
                            # lattice into 4 sublattices that we update one after the other
                            for ix in 1:NX
                                site = SiteCoords(ix, iy, iz, it)
                                if mod1(it+iz+iy+ix + site[μ], 4)!=pass
                                    continue
                                end
                                threadlocal += f!(U, μ, site, args...)
                            end
                        end
                    end
                end
                out += sum(threadlocal)
            end
        end
    end

    return out
end

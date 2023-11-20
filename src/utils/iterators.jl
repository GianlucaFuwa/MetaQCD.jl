
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
        @batch per=thread for site in eachindex(U)
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

    for _ in 1:count
        for μ in 1:4
            for pass in 1:4
                for it in 1:NT
                    for iz in 1:NZ
                        for iy in 1:NY
                            for ix in 1+mod(it + iz + iy, pass):4:NX
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

function sweep!(::Checkerboard4MT, count, f!, U, args...)
    NX, NY, NZ, NT = size(U)

    for _ in 1:count
        for μ in 1:4
            for pass in 1:4
                @threads for it in 1:NT
                    for iz in 1:NZ
                        for iy in 1:NY
                            for ix in 1+mod(it + iz + iy, pass):4:NX
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
    for _ in 1:count
        @batch per=thread threadlocal=0.0::Float64 for site in eachindex(U)
            for μ in 1:4
                threadlocal += f!(U, μ, site, args...)
            end
        end
    end

    return sum(threadlocal)
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
                            for ix in 1+mod(it + iz + iy, pass):4:NX
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

function sweep_reduce!(::Checkerboard4MT, count, f!, U, args...)
    NX, NY, NZ, NT = size(U)
    out = 0.0

    for _ in 1:count
        for μ in 1:4
            for pass in 1:4
                @batch per=thread threadlocal=0.0::Float64 for it in 1:NT
                    for iz in 1:NZ
                        for iy in 1:NY
                            for ix in 1+mod(it + iz + iy, pass):4:NX
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

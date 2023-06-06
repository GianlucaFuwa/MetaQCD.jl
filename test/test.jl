#include("../src/system/MetaQCD.jl")
#using .MetaQCD
# using .MetaQCD.Utils
using LinearAlgebra
using Polyester
using Base.Threads
using Random
using StaticArrays

foreach(i -> Random.seed!(Random.default_rng(i), i), 1:Threads.nthreads())

NX = 4; NY = 4; NZ = 4; NT = 4
action = WilsonGaugeAction
U = random_gauges(NX, NY, NZ, NT, 5.7, type_of_gaction = action)
#filename = "./test/testconf.txt"
#load_BridgeText!(filename,U)

function plaq_trace_sum_threads(u::T) where {T<:Gaugefield}
    NX, NY, NZ, NT = size(u)
    plaq = zeros(Float64, nthreads() * 8)

    @threads for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz ,it)
                    for μ in 1:3
                        for ν in μ+1:4
                            plaq[threadid() * 8] += real(tr(plaquette(u, μ, ν, site)))
                        end
                    end
                end	
            end
        end
    end

    return sum(plaq)
end

function plaq_trace_sum_batch(u::T) where {T<:Gaugefield}
    NX, NY, NZ, NT = size(u)
    plaq = zeros(Float64, nthreads() * 8)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz ,it)
                    for μ in 1:3
                        for ν in μ+1:4
                            plaq[threadid() * 8] += real(tr(plaquette(u, μ, ν, site)))
                        end
                    end  
                end	
            end
        end
    end

    return sum(plaq)
end

function plaq_trace_sum_site(u::T) where {T<:Gaugefield}
    NX, NY, NZ, NT = size(u)
    plaq = 0.0

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)
                    for μ in 1:3
                        for ν in μ+1:4
                            plaq += real(tr(plaquette(u, μ, ν, site)))
                        end
                    end  
                end	
            end
        end
    end

    return plaq
end

function staple_eachsite_s!(staples, U::Gaugefield{T}) where {T}
    NX, NY, NZ, NT = size(U)

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    for μ in 1:4
                        staples[μ][site] = staple(U, μ, site)
                    end

                end
            end
        end
    end

    return nothing
end

function staple_eachsite!(staples, U::Gaugefield{T}) where {T}
    NX, NY, NZ, NT = size(U)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    for μ in 1:4
                        staples[μ][ix, iy, iz, it] = staple(U, μ, site)
                    end

                end
            end
        end
    end

    return nothing
end

function staple_eachsite_eo!(staples, U::Gaugefield{T}) where {T}
    NX, NY, NZ, NT = size(U)

    for eo in 0:1
        @batch for it in 1:NT
            for iz in 1:NZ
                for iy in 1:NY
                    for ix in 1+eo:2:NX
                        site = SiteCoords(ix, iy, iz, it)
                        for μ in 1:4
                            staples[μ][site] = staple(U, μ, site)
                        end
                    end
                end
            end
        end
    end

    return nothing
end
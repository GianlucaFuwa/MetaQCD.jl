using MetaQCD
using FLoops
using LinearAlgebra
using Polyester
using Base.Threads: @threads, nthreads, threadid

L = (12, 12, 12, 12)
β = 6.0
gaction = WilsonGaugeAction

U = MetaQCD.initial_gauges("hot", L..., β, gaction);

function plaquette_trace_sum_threads(U)
    out = zeros(Float64, 8nthreads())

    @threads for site in eachindex(U)
        for μ in 1:3
            for ν in μ+1:4
                out[8threadid()] += real(tr(plaquette(U, μ, ν, site)))
            end
        end
    end

    return sum(out)
end

function plaquette_trace_sum_batch(U)
    out = zeros(Float64, 8nthreads())

    @batch for site in eachindex(U)
        for μ in 1:3
            for ν in μ+1:4
                p = real(tr(plaquette(U, μ, ν, site)))
                out[8threadid()] += p
            end
        end
    end

    return sum(out)
end

function plaquette_trace_sum_floop(U)::Float64
    @floop for site in eachindex(U)
        for μ in 1:3
            for ν in μ+1:4
                p = real(tr(plaquette(U, μ, ν, site)))
                @reduce(out = zero(Float64) + p)
            end
        end
    end
    out::Float64 = convert(Float64, out)
    return out
end

const out = zeros(Float64, 8nthreads())

function test_setindex(U)
    # out = 0.0
    for site in eachindex(U)
        for μ in 1:4
            ix, iy, iz, it = site.I
            i = 8threadid()
            out[i] += real(tr(U[μ][ix,iy,iz,it]))
        end
    end
    res = sum(out)
    out .= 0.0
    return res
end

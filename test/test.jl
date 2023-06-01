# include("../src/system/MetaQCD.jl")
# using .MetaQCD
using LinearAlgebra
using Polyester
using Base.Threads
using Random
Random.seed!(1206)

NX = 8; NY = 8; NZ = 8; NT = 8
action = WilsonGaugeAction
U = random_gauges(NX, NY, NZ, NT, 5.7, type_of_gaction = action)
#filename = "./test/testconf.txt"
#load_BridgeText!(filename,U)

function trace_sum(u::T) where {T<:Gaugefield}
    NX, NY, NZ, NT = size(u)
    plaq = zeros(Float64, nthreads() * 8)

    @threads for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        plaq[threadid() * 8] += real(tr(u[μ][ix,iy,iz,it])) +
                        real(tr(u[2][ix,iy,iz,it]))
                    end  
                end	
            end
        end
    end

    return sum(plaq)
end

function trace_sum_1(u::T) where {T<:Gaugefield}
    NX, NY, NZ, NT = size(u)
    plaq = zeros(Float64, nthreads() * 8)

    for μ in 1:4
        @threads for it in 1:NT
            for iz in 1:NZ
                for iy in 1:NY
                    for ix in 1:NX
                        plaq[threadid() * 8] += real(tr(u[μ][ix,iy,iz,it])) +
                        real(tr(u[2][ix,iy,iz,it]))
                    end	
                end
            end
        end
    end

    return sum(plaq)
end
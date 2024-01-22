using Accessors
using StaticArrays
using MetaQCD.Utils
using LinearAlgebra
using BenchmarkTools

remultr(args...) = real(tr(*(args...)))

U_v = Vector{Array{SMatrix{3,3,ComplexF64,9},4}}(undef, 4)
for μ in 1:4
    U_v[μ] = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, 4, 4, 4, 4)
    fill!(U_v[μ], eye3)
end

U_a = OffsetArray{SMatrix{3,3,ComplexF64,9},5}(undef, 0:5, 0:5, 0:5, 0:5, 0:5)
fill!(U_a, eye3)

function plaquette_sum(U::Vector{Array{SMatrix{3,3,ComplexF64,9},4}})
    p = zeros(Float64, 8Threads.nthreads())

    for site in CartesianIndices((4, 4, 4, 4))
        for μ in 1:3
            for ν in μ+1:4
                p[8Threads.threadid()] += plaquette(U, μ, ν, site)
            end
        end
    end

    return p
end

function plaquette_sum(U::OffsetArray{SMatrix{3,3,ComplexF64,9},5})
    p = zeros(Float64, 8Threads.nthreads())

    for site in CartesianIndices((4, 4, 4, 4))
        for μ in 1:3
            for ν in μ+1:4
                p[8Threads.threadid()] += plaquette(U, μ, ν, site)
            end
        end
    end

    return p
end

@inline function plaquette(U::Array{SMatrix{3,3,ComplexF64,9},5}, μ, ν, site)
    siteμ⁺ = move(site, μ, 1, 4)
    siteν⁺ = move(site, ν, 1, 4)
    return remultr(U[μ,site], U[ν,siteμ⁺], U[μ,siteν⁺], U[ν,site])
end

@inline function plaquette(U::Vector{Array{SMatrix{3,3,ComplexF64,9},4}}, μ, ν, site)
    siteμ⁺ = move(site, μ, 1, 4)
    siteν⁺ = move(site, ν, 1, 4)
    return remultr(U[μ][site], U[ν][siteμ⁺], U[μ][siteν⁺], U[ν][site])
end

# @benchmark plaquette_sum($U_v)
# @benchmark plaquette_sum($U_a)

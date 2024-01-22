using Accessors
using Revise
using ChainRulesCore
using Enzyme
using LinearAlgebra
using MetaQCD.Utils
using Random
using StaticArrays
using Zygote
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules

include("algebra.jl")

# Random.seed!(1206)
# U = Array{SMatrix{3, 3, ComplexF64, 9}, 5}(undef, 4, 4, 4, 4, 4); fill!(U, eye3);
# dU = similar(U); fill!(dU, zero3);
# U1 = initial_gauges("hot", 4, 4, 4, 4, 6.0; type_of_gaction=WilsonGaugeAction);
# dU1 = Temporaryfield(U1);
# staples = Temporaryfield(U1);

# for μ in 1:4
#     view(U, μ, :, :, :, :) .= U1[μ]
# end

function ChainRulesCore.rrule(::typeof(remultr), args::Vararg{Union{Daggered{T},T}, N}) where {N,T}
    y = remultr(args...)
    function remultr_pullback(ȳ)
        f̄ = NoTangent()
        tangents = ntuple(i ->
            @thunk(ifelse(args[i] isa Daggered, -1, 1)*0.5traceless_antihermitian(*ᶜ(circshift(i, args...)...))), Val(N))
    return f̄, tangents...
    end
    return y, remultr_pullback
end

function ChainRulesCore.rrule(::typeof(plaquette_sum),
                              U::Array{SMatrix{3, 3, ComplexF64, 9}, 5})
    y = plaquette_sum(U)
    function plaquette_sum_pullback(ȳ)
        Ū = zero(U)
        Ū .+= (ȳ,)
        return NoTangent(), Ū
    end
    y, plaquette_sum_pullback
end

function plaquette_sum(U)
    p = 0.0

    for site in CartesianIndices(size(U)[2:end])
        for μ in 1:3
            for ν in μ+1:4
                p += plaquette(U, μ, ν, site)
            end
        end
    end

    return p
end

@inline function plaquette(U, μ, ν, site)
    Nμ = size(U)[1+μ]
    Nν = size(U)[1+ν]
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteν⁺ = move(site, ν, 1, Nν)
    return remultr(U[μ,site], U[ν,siteμ⁺], U[μ,siteν⁺], U[ν,site])
end

matrices = Union{Daggered{SMatrix{3, 3, ComplexF64, 9}}, SMatrix{3, 3, ComplexF64, 9}}[]
for i in 1:4
    mat = iseven(i) ? gen_SU3_matrix(0.5) : Daggered(gen_SU3_matrix(0.5))
    push!(matrices, mat)
end

remultr(matrices...)
dTr = Zygote.gradient(remultr, matrices...);
plaquette_sum(U)
dP = Zygote.gradient(plaquette_sum, U);

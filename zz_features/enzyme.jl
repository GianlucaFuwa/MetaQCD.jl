using Revise
using Accessors
using BenchmarkTools
using Cthulhu
using Enzyme
using LinearAlgebra
using LoopVectorization
using MetaQCD
using MetaQCD.Utils
using Polyester
using Random
using StaticArrays
using Base.Threads
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules

include("algebra.jl")

Random.seed!(1206)
EnzymeRules.inactive(::typeof(move), args...) = nothing

U = Array{SMatrix{3, 3, ComplexF64, 9}, 5}(undef, 4, 4, 4, 4, 4); fill!(U, eye3);
dU = similar(U); fill!(dU, zero3);
U1 = initial_gauges("hot", 4, 4, 4, 4, 6.0; type_of_gaction=WilsonGaugeAction);
dU1 = Temporaryfield(U1);
staples = Temporaryfield(U1);

for μ in 1:4
    view(U, μ, :, :, :, :) .= U1[μ]
end

function wilson_gauge_action(U)
    P = plaquette_sum(U)
    Sg_wilson = 6.0 * (6*prod(size(U)[2:end]) - 1/3*P)
    return Sg_wilson
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

### RULES FOR REMULTR ###
function augmented_primal(config::ConfigWidth{1}, func::Const{typeof(remultr)},
    ::Type{<:Active}, args::Vararg{Active,N}) where {N}
    argvals = ntuple(i -> args[i].val, Val(N))
    if needs_primal(config)
        primal = func.val(argvals...)
    else
        primal = nothing
    end
    if overwritten(config)[3]
        tape = copy(argvals)
    else
        tape = nothing
    end

    return AugmentedReturn(primal, nothing, tape)
end

function EnzymeRules.reverse(config::ConfigWidth{1}, func::Const{typeof(remultr)},
    dret::Active, tape, args::Vararg{Active,N}) where {N}

    argvals = ntuple(i -> args[i].val, Val(N))
    dargs = ntuple(Val(N)) do i
        # is_dagg = args[i].val isa Daggered
        # sgn = ifelse(is_dagg, -1, 1)
        # mat = sgn*0.5traceless_antihermitian(*ᶜ(circshift(i-1, argvals...)...))
        # is_dagg ? mat' : mat
        0.5traceless_antihermitian(*ᶜ(circshift(i-1, argvals...)...))
    end
    return dargs
end

function Enzyme.gradient(::ReverseMode, f::typeof(remultr),
    args::Vararg{Union{T,Daggered{T}},N}) where {N,T}
    annots = ntuple(i -> Active(args[i]), Val(N))
    der = autodiff(Reverse, f, Active, annots...)
    return der
end

function Enzyme.gradient!(::ReverseMode, dU::Array{SMatrix{3,3,ComplexF64,9},5},
    f::typeof(wilson_gauge_action), U::Array{SMatrix{3,3,ComplexF64,9},5})
    autodiff(Reverse, f, Active, DuplicatedNoNeed(U, dU))
    return nothing
end

# matrices = Union{Daggered{SMatrix{3, 3, ComplexF64, 9}}, SMatrix{3, 3, ComplexF64, 9}}[]
# for i in 1:4
#     mat = iseven(i) ? Daggered(gen_SU3_matrix(0.5)) : gen_SU3_matrix(0.5)
#     push!(matrices, mat)
# end
# remultr(matrices...)
# Enzyme.gradient(Reverse, remultr, matrices...)

dU = zero(U);
Enzyme.gradient!(Reverse, dU, wilson_gauge_action, U)
dU[1,1,1,1,1]

# # dU1 = Temporaryfield(U1);
# MetaQCD.Updates.calc_dSdU!(dU1, staples, U1)
# dU1[1][1,1,1,1]

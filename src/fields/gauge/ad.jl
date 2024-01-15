using Revise
using Accessors
using BenchmarkTools
using Cthulhu
using Enzyme
using .EnzymeRules
using LinearAlgebra
using LoopVectorization
using MetaQCD
using MetaQCD.Utils
using Polyester
using Random
using StaticArrays
using Base.Threads
import .EnzymeRules: forward, reverse, augmented_primal

Random.seed!(1206)
EnzymeRules.inactive(::typeof(move), args...) = nothing

# function forward(func::Const{typeof(multr)}, ::Type{<:Duplicated}, y::Duplicated,
#                  x::Duplicated)
#     println("Using multr rule!")
#     ret = func.val(y.val, x.val)
#     y = @set y.dval = 0.5im*cmatmul_oo(y.val, x.val)
#     @show y.val
#     @show x.val
#     @show ret
#     return Duplicated(ret, multr(y.dval, x.val))
# end

# Y = gen_SU3_matrix(0.5)
# DY = zero3
# X = gen_SU3_matrix(0.5)
# DX = @SMatrix ones(ComplexF64, 3, 3)
# DY
# @show autodiff(Forward, multr, Duplicated(Y, DY), Duplicated(X, DX));

# A = SVector(1., 2., 3.); B = SVector(2., 2., 2.)

# function Enzyme.gradient(::ReverseMode, f::F, x::SMatrix{N,N,Complex{T},N²}) where {F,N,N²,T}
#     dx = MMatrix(zero(x))
#     autodiff(Reverse, f, Duplicated(MMatrix(x), dx))
#     SMatrix(dx)
# end
# @btime gradient(Reverse, sum, $X)

U = Array{SMatrix{3, 3, ComplexF64, 9}, 5}(undef, 4, 12, 12, 12, 12); fill!(U, eye3);
dU = similar(U); fill!(dU, zero3);
U1 = initial_gauges("hot", 12, 12, 12, 12, 6.0; type_of_gaction=WilsonGaugeAction);
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
    p = zeros(Float64, 8nthreads())

    @threads for site in CartesianIndices(size(U)[2:end])
        for μ in 1:3
            for ν in μ+1:4
                Nμ = size(U)[1+μ]
                Nν = size(U)[1+ν]
                siteμ⁺ = move(site, μ, 1, Nμ)
                siteν⁺ = move(site, ν, 1, Nν)
                plaq = cmatmul_oodd(U[μ,site], U[ν,siteμ⁺], U[μ,siteν⁺], U[ν,site])
                p[8threadid()] += real(tr(plaq))
            end
        end
    end

    return sum(p)
end

function Enzyme.gradient!(::ReverseMode, dx::Array{SMatrix{3, 3, ComplexF64, 9}, 5},
                         f::typeof(wilson_gauge_action), x::Array{SMatrix{3, 3, ComplexF64, 9}, 5})
    autodiff(Reverse, f, Active, DuplicatedNoNeed(x, dx))
    nothing
end

calc_gauge_action(U1)
wilson_gauge_action(U)

gradient!(Reverse, dU, wilson_gauge_action, U)
MetaQCD.Updates.calc_dSdU!(dU1, staples, U1)

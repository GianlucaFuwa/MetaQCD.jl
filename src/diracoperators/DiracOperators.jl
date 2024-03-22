module DiracOperators

using CUDA: i32
using KernelAbstractions # With this we can write generic GPU kernels for ROC and CUDA
using LinearAlgebra
using Polyester
using StaticArrays
using ..CG
using ..Output
using ..Utils

import ..Gaugefields: Fermionfield, Gaugefield, dims

abstract type AbstractDiracOperator end
abstract type AbstractFermionAction end

get_cg_temps(action::AbstractFermionAction) = action.temp1, action.temp2, action.temp3

"""
    Daggered(D::AbstractDiracOperator)

Wrap the Dirac operator `D` such that future functions know to treat it as `D†`
"""
struct Daggered{T} <: AbstractDiracOperator
    parent::T
    Daggered(D::T) where {T<:AbstractDiracOperator} = new{T}(D)
end

"""
    Hermitian(D::AbstractDiracOperator)

Wrap the Dirac operator `D` such that future functions know to treat it as `DD†`
"""
struct Hermitian{T} <: AbstractDiracOperator
    parent::T
    Hermitian(D::T) where {T<:AbstractDiracOperator} = new{T}(D)
end

"""
    replace_U!(D, U)

Replace the gauge field that is referenced by `D` with `U`
"""
function replace_U!(D::AbstractDiracOperator, U::Gaugefield)
    D.U = U
    return nothing
end

function replace_U!(D::Union{Daggered,Hermitian}, U::Gaugefield)
    D.parent.U = U
    return nothing
end

"""
    solve_D⁻¹x!(ϕ, D, ψ, temps...)

Solve the equation `Dϕ = ψ` for `ϕ`, where `D` is a Dirac operator
and store the result in `ϕ`. The `temps` argument is a list of temporary fields that
are used to store intermediate results.
"""
function solve_D⁻¹x!(ϕ, D, ψ, temps...)
    @assert dims(ϕ) == dims(ψ)
    cg!(ϕ, ψ, D, temps...)
    return nothing
end

include("staggered.jl")
include("wilson.jl")

end

module DiracOperators

using CUDA: i32
using KernelAbstractions # With this we can write generic GPU kernels for ROC and CUDA
using LinearAlgebra
using Polyester
using StaticArrays
using ..CG
using ..Output
using ..Utils

import ..Gaugefields: Fermionfield, Gaugefield, clear!, dims

abstract type AbstractDiracOperator end
abstract type AbstractFermionAction end

# So we don't print the entire array in the REPL...
function Base.show(io::IO, ::MIME"text/plain", D::T) where {T<:AbstractDiracOperator}
    print(io, "$(typeof(D))", "(;")
    for fieldname in fieldnames(T)
        if fieldname ∈ (:U, :temp)
            continue
        else
            print(io, " ", fieldname, " = ", getfield(D, fieldname), ",")
        end
    end
    print(io, ")")
    return nothing
end

function get_cg_temps(action::AbstractFermionAction)
    return action.temp1, action.temp2, action.temp3, action.temp4
end

function boundary_factor(anti, it, dir, NT)
    if !anti
        return 1
    else
        if dir == 1
            return it == NT ? -1 : 1
        elseif dir == -1
            return it == 1 ? -1 : 1
        else
            return 1
        end
    end
end

"""
    Daggered(D::AbstractDiracOperator)

Wrap the Dirac operator `D` such that future functions know to treat it as `D†`
"""
struct Daggered{T} <: AbstractDiracOperator
    parent::T
    Daggered(D::T) where {T<:AbstractDiracOperator} = new{T}(D)
end

LinearAlgebra.adjoint(D::AbstractDiracOperator) = Daggered(D)

"""
    DdaggerD(D::AbstractDiracOperator)

Wrap the Dirac operator `D` such that future functions know to treat it as `D†D`
"""
struct DdaggerD{T} <: AbstractDiracOperator
    parent::T
    DdaggerD(D::T) where {T<:AbstractDiracOperator} = new{T}(D)
end

"""
    replace_U!(D, U)

Replace the gauge field that is referenced by `D` with `U`
"""
function replace_U!(D::AbstractDiracOperator, U::Gaugefield)
    D.U = U
    return nothing
end

function replace_U!(D::Union{Daggered,DdaggerD}, U::Gaugefield)
    D.parent.U = U
    return nothing
end

"""
    solve_D⁻¹x!(ϕ, D, ψ, temps...)

Solve the equation `Dϕ = ψ` for `ϕ`, where `D` is a Dirac operator
and store the result in `ϕ`. The `temps` argument is a list of temporary fields that
are used to store intermediate results.
"""
function solve_D⁻¹x!(ϕ, D::T, ψ, temps...; tol=1e-10, maxiters=1000) where {T<:DdaggerD}
    @assert dims(ϕ) == dims(ψ)
    cg!(ϕ, ψ, D, temps...; tol=tol, maxiters=maxiters)
    return nothing
end

"""
    calc_fermion_action(fermion_action, U, ψ)

Calculate the fermion action for the fermion field `ψ` on the gauge background `U`
using the fermion action `fermion_action`
"""
function calc_fermion_action(fermion_action::AbstractFermionAction, U::Gaugefield, ψ::Fermionfield)
    replace_U!(fermion_action.D, U)
    return calc_fermion_action(fermion_action, ψ)
end

include("staggered.jl")
include("wilson.jl")

end

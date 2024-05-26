module DiracOperators

using KernelAbstractions # With this we can write generic GPU kernels for ROC and CUDA
using LinearAlgebra
using Polyester
using StaticArrays
using ..CG
using ..Output
using ..Utils

import ..Gaugefields: Abstractfield, EvenOdd, Fermionfield, Gaugefield, Tensorfield, clear!
import ..Gaugefields: check_dims, clover_square, dims, even_odd, gaussian_pseudofermions!
import ..Gaugefields: Checkerboard2, Sequential, @latmap, @latsum, set_source!
import ..RHMCParameters: RHMCParams

abstract type AbstractDiracOperator end
abstract type AbstractFermionAction end

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
    solve_dirac!(ψ, D, ϕ, temp1, temp2, temp3, tol=1e-16, maxiters=1000)

Solve the Dirac equation `Dψ = ϕ` for `ψ`, where `D` is a Dirac operator and store the
result in `ψ`.

The methods unique to each dirac operator definition are contained within their respective
files.
"""
function solve_dirac!(
    ψ, D::T, ϕ, temp1, temp2, temp3, tol=1e-16, maxiters=1000
) where {T<:DdaggerD}
    cg!(ψ, D, ϕ, temp1, temp2, temp3; tol=tol, maxiters=maxiters)
    return nothing
end

"""
    solve_dirac_multishift!(ψs, shifts, D, ϕ, temps...)

Solve the equations `(D + s)ψ = ϕ` for `ψ` for each `s` in `shifts`, where `D` is a
Dirac operator and store each result in `ψs`.

The methods unique to each dirac operator definition are contained within their respective
files.
"""
function solve_dirac_multishift!(
    ψs, shifts, D::T, ϕ, temp1, temp2, ps, tol=1e-16, maxiters=1000
) where {T<:DdaggerD}
    mscg!(ψs, shifts, D, ϕ, temp1, temp2, ps; tol=tol, maxiters=maxiters)
    return nothing
end

"""
    calc_fermion_action(fermion_action, ϕ)

Calculate the fermion action for the fermion field `ϕ` using the fermion action
`fermion_action`
"""
function calc_fermion_action(fermion_action::TA, U, ϕ::TF) where {TA,TF}
    @nospecialize fermion_action U ϕ
    return error("calc_fermion_action is not supported for type $TA with field of type $TF")
end

"""
    sample_pseudofermions!(ϕ, fermion_action)

Sample pseudo fermions for an HMC update according to the probability density specified by
`fermion_action`
"""
sample_pseudofermions!(::Nothing, ::Nothing) = nothing

@inline function boundary_factor(anti, it, dir, NT)
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

include("staggered.jl")
include("staggered_eo.jl")
include("wilson.jl")
# include("wilson_eo.jl")
include("gpu_kernels/staggered.jl")
include("gpu_kernels/staggered_eo.jl")
include("gpu_kernels/wilson.jl")

function fermaction_from_str(str, eo_precon::Bool)
    if str == "wilson"
        return WilsonFermionAction
    elseif str == "staggered"
        return eo_precon ? StaggeredEOPreFermionAction : StaggeredFermionAction
    elseif str == "none" || str === nothing
        return nothing
    else
        error("fermion action \"$(str)\" not supported")
    end
end

const FERMION_ACTION = Dict{Tuple{String,Bool},Type{<:AbstractFermionAction}}(
    # boolean determines whether even-odd or not
    ("wilson", false) => WilsonFermionAction,
    ("staggered", false) => StaggeredFermionAction,
    ("staggered", true) => StaggeredEOPreFermionAction,
)

end

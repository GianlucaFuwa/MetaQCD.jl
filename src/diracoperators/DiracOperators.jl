"""
    module DiracOperators

This module's files are structured as follows:

Each different Dirac operator gets its own file, where it and its corresponding action
get their own structs. The files also contain `mul!` functions for the regular operator, its
adjoint and Hermitian (D†D convention) counterpart, which are used to make them act on
`Spinorfield`s. For the `mul!` we also define the kernels in the respective files.
"""
module DiracOperators

using LinearAlgebra: checksquare
using KernelAbstractions # With this we can write generic GPU kernels for ROC and CUDA
using LinearAlgebra
using Polyester
using SparseArrays
using StaticArrays
using ..MetaIO
using ..RHMCParameters
using ..Solvers
using ..Utils

import KernelAbstractions as KA
import ..Fields: AbstractField, Gaugefield, Spinorfield, SpinorfieldEO, Tensorfield
import ..Fields: check_dims, clear!, clover_square, dims, even_odd, gaussian_pseudofermions!
import ..Fields: Clover, Checkerboard2, Sequential, @latmap, @latsum, set_source!, volume
import ..Fields: fieldstrength_eachsite!, fieldstrength_A_eachsite!, num_colors, num_dirac
import ..Fields: PeriodicBC, AntiPeriodicBC, apply_bc, create_bc, distributed_reduce

abstract type AbstractDiracOperator end
abstract type AbstractFermionAction{Nf} end

struct QuenchedFermionAction <: AbstractFermionAction{0} end

# some aliases
const StaggeredSpinorfield{B,T,M,A} = Spinorfield{B,T,M,A,1}
const StaggeredEOPreSpinorfield{B,T,M,A} = SpinorfieldEO{B,T,M,A,1}
const WilsonSpinorfield{B,T,M,A} = Spinorfield{B,T,M,A,4}
const WilsonEOPreSpinorfield{B,T,M,A} = SpinorfieldEO{B,T,M,A,4}

Base.eltype(D::AbstractDiracOperator) = eltype(D.temp)
LinearAlgebra.checksquare(D::AbstractDiracOperator) = LinearAlgebra.checksquare(D.temp)
get_temp(D::AbstractDiracOperator) = D.temp
@inline num_flavors(::AbstractFermionAction{Nf}) where {Nf} = Nf

"""
    Daggered(D::AbstractDiracOperator)

Wrap the Dirac operator `D` such that future functions know to treat it as `D†`
"""
struct Daggered{T} <: AbstractDiracOperator
    parent::T
    Daggered(D::T) where {T<:AbstractDiracOperator} = new{T}(D)
end

LinearAlgebra.adjoint(D::AbstractDiracOperator) = Daggered(D)
LinearAlgebra.checksquare(D::Daggered) = LinearAlgebra.checksquare(D.parent)
Base.eltype(D::Daggered) = eltype(D.parent)
get_temp(D::Daggered) = D.parent.temp

"""
    DdaggerD(D::AbstractDiracOperator)

Wrap the Dirac operator `D` such that future functions know to treat it as `D†D`
"""
struct DdaggerD{T} <: AbstractDiracOperator
    parent::T
    DdaggerD(D::T) where {T<:AbstractDiracOperator} = new{T}(D)
end

LinearAlgebra.checksquare(D::DdaggerD) = LinearAlgebra.checksquare(D.parent)
Base.eltype(D::DdaggerD) = eltype(D.parent)
get_temp(D::DdaggerD) = D.parent.temp

"""
    solve_dirac!(ψ, D, ϕ, temp1, temp2, temp3, tol=1e-16, maxiters=1000)

Solve the Dirac equation `Dψ = ϕ` for `ψ`, where `D` is a Hermitian Dirac operator and
store the result in `ψ`.
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
Hermitian Dirac operator and store each result in `ψs`.
"""
function solve_dirac_multishift!(
    ψs, shifts, D::T, ϕ, temp1, temp2, ps, tol=1e-16, maxiters=1000
) where {T<:DdaggerD}
    mscg!(ψs, SVector(shifts), D, ϕ, temp1, temp2, ps; tol=tol, maxiters=maxiters)
    return nothing
end

"""
    calc_fermion_action(fermion_action, ϕ)

Calculate the fermion action for the fermion field `ϕ` using the fermion action
`fermion_action`.
"""
function calc_fermion_action(fermion_action::TA, U, ϕ::TF) where {TA,TF}
    @nospecialize fermion_action U ϕ
    return error("calc_fermion_action is not supported for type $TA with field of type $TF")
end

calc_fermion_action(::QuenchedFermionAction, ::Gaugefield) = 0.0

"""
    sample_pseudofermions!(ϕ, fermion_action)

Sample pseudo fermions for an HMC update according to the probability density specified by
`fermion_action`.
"""
sample_pseudofermions!(::AbstractField, ::QuenchedFermionAction) = nothing

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

function Base.show(io::IO, D::T) where {T<:AbstractDiracOperator}
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

function construct_diracmatrix(D, U)
    n = checksquare(D)
    Du = D(U)
    M = spzeros(ComplexF64, n, n) 
    temp1 = similar(get_temp(D))
    temp2 = similar(get_temp(D))
    ND = num_dirac(temp1)
    fdims = dims(U)
    NV = U.NV
    @assert n < 5000
    is_evenodd = temp1 isa SpinorfieldEO
   
    ii = 1

    for isite in eachindex(U)
        if is_evenodd
            iseven(isite) || continue
        end

        for α in 1:ND
            for a in 1:3
                set_source!(temp1, isite, a, α)
                mul!(temp2, Du, temp1)
                jj = 1

                for jsite in eachindex(U)
                    if is_evenodd
                        iseven(jsite) || continue
                        _jsite = eo_site(jsite, fdims..., NV)
                    else
                        _jsite = jsite
                    end

                    for β in 1:ND
                        for b in 1:3
                            ind = (β - 1) * 3 + b
                            M[jj, ii] = temp2[_jsite][ind]
                            jj += 1
                        end
                    end
                end

                ii += 1
            end
        end
    end

    return M
end

include("staggered.jl")
include("staggered_eo.jl")
include("wilson.jl")
include("wilson_eo.jl")
include("gpu_kernels/staggered.jl")
include("gpu_kernels/staggered_eo.jl")
include("gpu_kernels/wilson.jl")
include("arnoldi.jl")

function fermaction_from_str(str, eo_precon::Bool)
    if str == "wilson"
        return eo_precon ? WilsonEOPreFermionAction : WilsonFermionAction
    elseif str == "staggered"
        return eo_precon ? StaggeredEOPreFermionAction : StaggeredFermionAction
    elseif str ∈ ("none", "quenched") || str === nothing
        return QuenchedFermionAction
    else
        error("fermion action \"$(str)\" not supported")
    end
end

const FERMION_ACTION = Dict{Tuple{String,Bool},Type{<:AbstractFermionAction}}(
    # boolean determines whether even-odd or not
    ("wilson", false) => WilsonFermionAction,
    ("wilson", true) => WilsonEOPreFermionAction,
    ("staggered", false) => StaggeredFermionAction,
    ("staggered", true) => StaggeredEOPreFermionAction,
)

function Base.show(io::IO, ::MIME"text/plain", S::AbstractFermionAction{Nf}) where {Nf}
    name = nameof(typeof(S))
    print(
        io,
        """
        
        |  $(name)(
        |    Nf = $Nf
        |    MASS = $(S.D.mass)
        """
    )

    if S isa WilsonFermionAction #|| S isa WilsonEOPreFermionAction
        print(
            io,
            """
            |    KAPPA: $(S.D.κ)
            |    CSW: $(S.D.csw)
            """
        )
    end

    print(
        io,
        """
        |    BOUNDARY CONDITION (TIME): $(S.D.boundary_condition))
        |    CG TOLERANCE (ACTION) = $(S.cg_tol_action)
        |    CG TOLERANCE (MD) = $(S.cg_tol_md)
        |    CG MAX ITERS (ACTION) = $(S.cg_maxiters_action)
        |    CG MAX ITERS (ACTION) = $(S.cg_maxiters_md)
        |    RHMC INFO (Action): $(S.rhmc_info_action)
        |    RHMC INFO (MD): $(S.rhmc_info_md))
        """
    )
    return nothing
end

function Base.show(io::IO, S::AbstractFermionAction{Nf}) where {Nf}
    name = nameof(typeof(S))
    print(
        io,
        """
        
        |  $(name)(
        |    Nf = $Nf
        |    MASS = $(S.D.mass)
        """
    )

    if S isa WilsonFermionAction #|| S isa WilsonEOPreFermionAction
        print(
            io,
            """
            |    KAPPA: $(S.D.κ)
            |    CSW: $(S.D.csw)
            """
        )
    end

    print(
        io,
        """
        |    BOUNDARY CONDITION (TIME): $(S.D.boundary_condition))
        |    CG TOLERANCE (ACTION) = $(S.cg_tol_action)
        |    CG TOLERANCE (MD) = $(S.cg_tol_md)
        |    CG MAX ITERS (ACTION) = $(S.cg_maxiters_action)
        |    CG MAX ITERS (ACTION) = $(S.cg_maxiters_md)
        |    RHMC INFO (Action): $(S.rhmc_info_action)
        |    RHMC INFO (MD): $(S.rhmc_info_md))
        """
    )
    return nothing
end

end

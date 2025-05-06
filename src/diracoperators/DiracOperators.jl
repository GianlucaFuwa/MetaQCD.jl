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
using Printf
using SparseArrays
using StaticArrays
using StaticTools: StaticString
using ..MetaIO
using ..RHMCParameters
using ..Solvers
using ..Utils

import KernelAbstractions as KA
import ..Fields: AbstractField, FieldTopology, Gaugefield, Paulifield, Spinorfield
import ..Fields: MultiSpinorfield, SpinorfieldEO, Tensorfield, num_spinors
import ..Fields: check_dims, clear!, clover_square, dims, even_odd, gaussian_pseudofermions!
import ..Fields: @latmap, @latsum, Clover, Checkerboard2, Sequential, set_source!, volume
import ..Fields: @groupreduce, fieldstrength_eachsite!, num_colors, num_dirac
import ..Fields: PeriodicBC, AntiPeriodicBC, apply_bc, create_bc, distributed_reduce

abstract type AbstractDiracOperator{B,T} end
abstract type AbstractFermionAction{R,Nf} end # R indicates whether the action uses rational approximation or not

struct QuenchedFermionAction <: AbstractFermionAction{false,0}
    QuenchedFermionAction(args...; kwargs...) = new()
end

# some aliases
const StaggeredSpinorfield{B,T,M,A} = Spinorfield{B,T,M,A,1}
const StaggeredEOPreSpinorfield{B,T,M,A} = SpinorfieldEO{B,T,M,A,1}
const WilsonSpinorfield{B,T,M,A} = Spinorfield{B,T,M,A,4}
const WilsonEOPreSpinorfield{B,T,M,A} = SpinorfieldEO{B,T,M,A,4}

Base.eltype(D::AbstractDiracOperator) = eltype(D.temp)
LinearAlgebra.checksquare(D::AbstractDiracOperator) = LinearAlgebra.checksquare(D.temp)
get_temp(D::AbstractDiracOperator) = D.temp
@inline num_flavors(::AbstractFermionAction{R,Nf}) where {R,Nf} = Nf

# To add gauge background to Dirac operator, apply it to a gaugefield
add_gauge_background(::TD, ::TG) where {TD,TG} = error("cannot set background $TG to $TD")
(D::AbstractDiracOperator)(U::Gaugefield) = add_gauge_background(D, U)

"""
    Daggered(D::AbstractDiracOperator)

Wrap the Dirac operator `D` such that future functions know to treat it as `D†`
"""
struct Daggered{TD,B,T} <: AbstractDiracOperator{B,T}
    parent::TD
    Daggered(D::TD) where {B,T,TD<:AbstractDiracOperator{B,T}} = new{TD,B,T}(D)
end

LinearAlgebra.adjoint(D::AbstractDiracOperator) = Daggered(D)
LinearAlgebra.checksquare(D::Daggered) = LinearAlgebra.checksquare(D.parent)
Base.eltype(D::Daggered) = eltype(D.parent)
get_temp(D::Daggered) = D.parent.temp

"""
    DdaggerD(D::AbstractDiracOperator)

Wrap the Dirac operator `D` such that future functions know to treat it as `D†D`
"""
struct DdaggerD{TD,B,T} <: AbstractDiracOperator{B,T}
    parent::TD
    DdaggerD(D::TD) where {B,T,TD<:AbstractDiracOperator{B,T}} = new{TD,B,T}(D)
end

LinearAlgebra.checksquare(D::DdaggerD) = LinearAlgebra.checksquare(D.parent)
Base.eltype(D::DdaggerD) = eltype(D.parent)
get_temp(D::DdaggerD) = D.parent.temp

include("staggered_eo.jl")
include("action.jl")
include("staggered.jl")
include("staggered_hoelbling.jl")
include("wilson.jl")
include("wilson_eo.jl")
include("gpu_kernels/staggered.jl")
include("gpu_kernels/staggered_eo.jl")
include("gpu_kernels/wilson.jl")
include("gpu_kernels/wilson_eo.jl")
include("arnoldi.jl")

const DIRAC_OPERATORS = Dict(
    "staggered" => StaggeredDiracOperator,
    "staggered_eo" => StaggeredEOPreDiracOperator,
    "staggered_h1234" => StaggeredHoelblingDiracOperator{1234},
    "staggered_h1342" => StaggeredHoelblingDiracOperator{1342},
    "wilson" => WilsonDiracOperator,
    "wilson_eo" => WilsonEOPreDiracOperator,
)

@inline function default_Nf(type)
    return if type == "staggered"
        8
    elseif type == "staggered_eo"
        4
    elseif type == "staggered_h1234"
        2
    elseif type == "staggered_h1342"
        2
    elseif type == "wilson"
        2
    elseif type == "wilson_eo"
        2
    else
        error("Fermion action type $(type) not supported")
    end
end

# need to be overloaded for all operators
function default_Nf(::AbstractDiracOperator)
    error("default_Nf not implemented for this type")
    return nothing
end

function is_staggered(::AbstractDiracOperator)
    error("is_staggered not implemented for this type")
    return nothing
end

# needs to be overloaded for operators that can contain a  clover term
function has_clover_term(::AbstractDiracOperator)
    error("has_clover_term not implemented for this type")
    return nothing
end

"""
    solve_dirac!(ψ, D, ϕ, temp1, temp2, temp3, tol=1e-16, maxiters=1000)

Solve the Dirac equation `Dψ = ϕ` for `ψ`, where `D` is a Hermitian Dirac operator and
store the result in `ψ`.
"""
function solve_dirac!(
    ψ, D::T, ϕ, temp1, temp2, temp3, tol=1e-16, maxiters=1000
) where {T<:DdaggerD}
    return cg!(ψ, D, ϕ, temp1, temp2, temp3; tol=tol, maxiters=maxiters)
end

"""
    solve_dirac_multishift!(ψs, shifts, D, ϕ, temps...)

Solve the equations `(D + s)ψ = ϕ` for `ψ` for each `s` in `shifts`, where `D` is a
Hermitian Dirac operator and store each result in `ψs`.
"""
function solve_dirac_multishift!(
    ψs, shifts, D::T, ϕ, temp1, temp2, ps, tol=1e-16, maxiters=1000
) where {T<:DdaggerD}
    return mscg!(ψs, SVector(shifts), D, ϕ, temp1, temp2, ps; tol=tol, maxiters=maxiters)
end

# So we don't print the entire array in the REPL...
function Base.show(io::IO, ::MIME"text/plain", D::T) where {T<:AbstractDiracOperator}
    print(io, "$(typeof(D))", "(;")

    for fieldname in fieldnames(T)
        if fieldname ∈ (:U, :temp, :D_diag, :D_oo_inv, :Fμν)
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

end

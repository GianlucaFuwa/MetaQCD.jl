module Forces

using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using StaticArrays
using Polyester: @batch
using Printf
using StaticTools: StaticString
using Unicode
using ..MetaIO
using ..RHMCParameters
using ..Utils

import ..BiasModule: calc_cv, calc_cv_deriv!, ∂V∂Q
import ..DiracOperators: StaggeredDiracOperator, StaggeredEOPreDiracOperator
import ..DiracOperators: StaggeredHoelblingDiracOperator, WilsonDiracOperator
import ..DiracOperators: WilsonEOPreDiracOperator, FermionAction, has_clover_term
import ..DiracOperators: Daggered, DdaggerD, Spinorfield, SpinorfieldEO, apply_bc
import ..DiracOperators: staggered_η, staggered_ημν, solve_dirac!, solve_dirac_multishift!
import ..DiracOperators: mul_oe!, mul_eo!, mul_oo_inv!, get_mass_term
import ..Fields: AbstractGaugeAction, Gaugefield, Colorfield, add!, global_dims
import ..Fields: allindices, clear!, dims, normalize!, fieldstrength_eachsite!, float_type
import ..Fields: check_dims, even_odd, mul!, staple, staple_eachsite!
import ..Fields: @groupreduce, @latmap, @latsum, gauge_action, is_distributed, update_halo!
import ..Fields: AbstractField, Plaquette, Clover, Spinorfield, Tensorfield
import ..Fields: Paulifield, MultiSpinorfield, gauge_action_deriv!
import ..Smearing: AbstractSmearing, NoSmearing, StoutSmearing
import ..Smearing: calc_smearedU!, get_layer, stout_backprop!

# some aliases
const StaggeredSpinorfield{B,T,M,A} = Spinorfield{B,T,M,A,1}
const StaggeredEOPreSpinorfield{B,T,M,A} = SpinorfieldEO{B,T,M,A,1}
const WilsonSpinorfield{B,T,M,A} = Spinorfield{B,T,M,A,4}
const WilsonEOPreSpinorfield{B,T,M,A} = SpinorfieldEO{B,T,M,A,4}

"""
    calc_dSfdU_bare!(dU::Colorfield, fermion_action, U, ϕ, ::Any, ::NoSmearing)
    calc_dSfdU_bare!(dU::Colorfield, fermion_action, U, ϕ, temp_force, smearing, is_smeared)

Calculate the derivative of `fermion_action` w.r.t. the gauge field `U` on the pseudofermion
background `ϕ` and store the result in `dU`.

If `temp_force isa Colorfield` and `bias.smearing != nothing`, the derivative is calculated
w.r.t. the fully smeared field V = 𝔉(U) using Stout smearing and Stout force recursion.

If `is_smeared = true`, it is assumed that smearing has already been applied to `U`,
meaning that the gauge fields in `smearing` are the smeared versions of `U`

# References

Daming Li "The Calulation of Force in Lattice Quantum Chromodynamics" (2022)
[doi:10.11648/j.ajpa.20221001.12](https://doi.org/10.11648/j.ajpa.20221001.12)
"""
function calc_dSfdU_bare!(dU, fermion_action, U, ϕ, ::Any, ::NoSmearing, ::Bool)
    calc_dSfdU!(dU, fermion_action, U, ϕ) # Defined in each operators respective file
    return nothing
end

function calc_dSfdU_bare!(dU, fermion_action, U, ϕ, temp_force, smearing, is_smeared=false)
    is_smeared || calc_smearedU!(smearing, U)
    fully_smeared_U = smearing.Usmeared_multi[end]
    calc_dSfdU!(dU, fermion_action, fully_smeared_U, ϕ) # Defined in each operators respective file
    stout_backprop!(dU, temp_force, smearing)
    return nothing
end

include("gauge_force.jl")
include("bias_force.jl")
include("wilson_force.jl")
include("wilson_eo_force.jl")
include("staggered_force.jl")
include("staggered_eo_force.jl")
include("staggered_hoelbling_force.jl")

include("gpu_kernels/wilson_force.jl")
include("gpu_kernels/staggered_force.jl")
include("gpu_kernels/staggered_hoelbling_force.jl")

end

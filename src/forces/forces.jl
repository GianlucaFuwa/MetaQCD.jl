# XXX: Maybe make this its own module?

# module Forces

import ..DiracOperators: StaggeredDiracOperator, StaggeredFermionAction
import ..DiracOperators: StaggeredEOPreDiracOperator, StaggeredEOPreFermionAction
import ..DiracOperators: WilsonDiracOperator, WilsonFermionAction
import ..DiracOperators: WilsonEOPreDiracOperator, WilsonEOPreFermionAction
import ..DiracOperators: WilsonEODiagonal
import ..DiracOperators: Daggered, DdaggerD, EvenOdd
import ..DiracOperators: boundary_factor, staggered_η, solve_dirac!, solve_dirac_multishift!
import ..DiracOperators: mul_oe!, mul_eo!, mul_oo_inv!

"""
    calc_dSfdU_bare!(dU::Colorfield, fermion_action, U, ϕ, ::Any, ::NoSmearing)
    calc_dSfdU_bare!(dU::Colorfield, fermion_action, U, ϕ, temp_force, smearing)

Calculate the derivative of `fermion_action` w.r.t. the gauge field `U` on the fermion
background `ϕ` and store the result in `dU`.
If `temp_force` and `smearing` are specified, the derivative is calculated w.r.t. the 
fully smeared field V = 𝔉(U) using Stout smearing and Stout force recursion.

# References

Daming Li "The Calulation of Force in Lattice Quantum Chromodynamics" (2022)
[doi:10.11648/j.ajpa.20221001.12](https://doi.org/10.11648/j.ajpa.20221001.12)
"""
function calc_dSfdU_bare!(dU, fermion_action, U, ϕ, ::Any, ::NoSmearing)
    calc_dSfdU!(dU, fermion_action, U, ϕ) # Defined in each operators respective file
    return nothing
end

function calc_dSfdU_bare!(dU, fermion_action, U, ϕ, temp_force, smearing)
    calc_smearedU!(smearing, U)
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

include("gpu_kernels/gauge_force.jl")
include("gpu_kernels/bias_force.jl")
include("gpu_kernels/wilson_force.jl")
include("gpu_kernels/staggered_force.jl")

# end

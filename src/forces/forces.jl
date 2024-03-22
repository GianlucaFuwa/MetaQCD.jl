import ..DiracOperators: StaggeredDiracOperator, StaggeredFermionAction, solve_D⁻¹x!
import ..DiracOperators: WilsonDiracOperator, WilsonFermionAction
import ..DiracOperators: get_cg_temps, staggered_η, solve_D⁻¹x!

"""
    calc_dSfdU!(dU, U, ψ, fermion_action)

Calculate the derivative of `fermion_action` w.r.t. the gauge field `U` on the fermion
background `ψ` and store the result in `dU`.

# References

Daming Li "The Calulation of Force in Lattice Quantum Chromodynamics" (2022)
[doi:10.11648/j.ajpa.20221001.12](https://doi.org/10.11648/j.ajpa.20221001.12)
"""
function calc_dSfdU! end

include("gauge_force.jl")
include("bias_force.jl")
include("wilson_force.jl")
include("staggered_force.jl")

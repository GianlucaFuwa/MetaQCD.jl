# Explicit precompilation including running code
using PrecompileTools: @setup_workload, @compile_workload

# @setup_workload begin
#     # Setup code can go here
#
#     @compile_workload begin
#         for T in (Float32, Float64)
#             for dirac in ("staggered", "staggered_eo", "wilson")
#                 U = Gaugefield{CPU,Float64,WilsonGaugeAction}(4, 4, 4, 4, 6.0)
#                 random_gauges!(U)
#                 eo_fun = contains(dirac, "eo") ? even_odd : identity
#                 ψ = eo_fun(Spinorfield(U; staggered=contains(dirac, "staggered")))
#                 action = FermionAction(
#                     dirac,
#                     U,
#                     0.01,
#                     Nf=1,
#                     bc_str="antiperiodic",
#                     rhmc_spectral_bound=(0.0001, 6.0),
#                     rhmc_order_action=15,
#                     cg_tol_action=1e-16,
#                     cg_maxiters_action=1000,
#                 )
#                 sample_pseudofermions!(ψ, action, U)
#                 calc_fermion_action(action, U, ψ)
#                 # Everything inside this block will run at precompile time, saving the
#                 # binary code to a cache in newer versions of Julia.
#             end
#         end
#     end
# end

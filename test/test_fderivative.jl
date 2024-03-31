using MetaQCD
using MetaQCD.Utils
using Random

const StaggeredFermionAction = MetaQCD.DiracOperators.StaggeredFermionAction
const WilsonFermionAction = MetaQCD.DiracOperators.WilsonFermionAction

function SU3testfderivative(backend=nothing)
    Random.seed!(123)
    println("SU3test_fermion_derivative")
    MetaQCD.Output.set_global_logger!(1, devnull; tc=false)
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    U = initial_gauges("cold", NX, NY, NZ, NT, 6.0; GA=WilsonGaugeAction)
    ψ_st = Fermionfield(NX, NY, NZ, NT; staggered=true)
    MetaQCD.Gaugefields.gaussian_pseudofermions!(ψ_st)
    ψ_wi = Fermionfield(NX, NY, NZ, NT)
    wilson_action = WilsonFermionAction(U, 1.0)
    MetaQCD.Gaugefields.gaussian_pseudofermions!(ψ_wi)
    staggered_action = StaggeredFermionAction(U, 1.0)
    # filename = "./test/testconf.txt"
    # loadU!(BridgeFormat(), U, filename);
    if backend !== nothing
        U = MetaQCD.to_backend(backend, U)
    end

    # Test for smearing with 5 steps and stout parameter 0.12
    smearing = StoutSmearing(U, 5, 0.12)

    dSfdU_st = Temporaryfield(U)
    dSfdU_st_smeared = Temporaryfield(U)
    dSfdU_wi = Temporaryfield(U)
    dSfdU_wi_smeared = Temporaryfield(U)
    temp_force = Temporaryfield(U)
    println("Initial action: ", calc_fermion_action(staggered_action, ψ_st), "\n")

    site = SiteCoords(2, 3, 1, 2)
    μ = 3
    ΔH = 0.000001

    relerrors = Matrix{Float64}(undef, 4, 8)

    for group_direction in 1:8
        # Unsmeared
        Ufwd = deepcopy(U)
        Ufwd[μ, site] = expλ(group_direction, ΔH) * Ufwd[μ, site]
        MetaQCD.DiracOperators.replace_U!(staggered_action.D, Ufwd)
        MetaQCD.DiracOperators.replace_U!(wilson_action.D, Ufwd)
        @assert Ufwd[μ, site] == staggered_action.D.U[μ, site]
        st_action_new_fwd = calc_fermion_action(staggered_action, ψ_st)
        wi_action_new_fwd = calc_fermion_action(wilson_action, ψ_wi)

        Ubwd = deepcopy(U)
        Ubwd[μ, site] = expλ(group_direction, -ΔH) * Ubwd[μ, site]
        MetaQCD.DiracOperators.replace_U!(staggered_action.D, Ubwd)
        MetaQCD.DiracOperators.replace_U!(wilson_action.D, Ubwd)
        st_action_new_bwd = calc_fermion_action(staggered_action, ψ_st)
        wi_action_new_bwd = calc_fermion_action(wilson_action, ψ_wi)

        # Smeared
        Ufwd = deepcopy(U)
        Ufwd[μ, site] = expλ(group_direction, ΔH) * Ufwd[μ, site]
        calc_smearedU!(smearing, Ufwd)
        MetaQCD.DiracOperators.replace_U!(staggered_action.D, smearing.Usmeared_multi[end])
        MetaQCD.DiracOperators.replace_U!(wilson_action.D, smearing.Usmeared_multi[end])
        st_action_new_fwd_smeared = calc_fermion_action(staggered_action, ψ_st)
        wi_action_new_fwd_smeared = calc_fermion_action(wilson_action, ψ_wi)

        Ubwd = deepcopy(U)
        Ubwd[μ, site] = expλ(group_direction, -ΔH) * Ubwd[μ, site]
        calc_smearedU!(smearing, Ubwd)
        MetaQCD.DiracOperators.replace_U!(staggered_action.D, smearing.Usmeared_multi[end])
        MetaQCD.DiracOperators.replace_U!(wilson_action.D, smearing.Usmeared_multi[end])
        st_action_new_bwd_smeared = calc_fermion_action(staggered_action, ψ_st)
        wi_action_new_bwd_smeared = calc_fermion_action(wilson_action, ψ_wi)

        MetaQCD.DiracOperators.replace_U!(staggered_action.D, U)
        MetaQCD.DiracOperators.replace_U!(wilson_action.D, U)
        calc_dSfdU_bare!(dSfdU_st, staggered_action, U, ψ_st, nothing, NoSmearing())
        calc_dSfdU_bare!(dSfdU_st_smeared, staggered_action, U, ψ_st, temp_force, smearing)
        calc_dSfdU_bare!(dSfdU_wi, wilson_action, U, ψ_wi, nothing, NoSmearing())
        calc_dSfdU_bare!(dSfdU_wi_smeared, wilson_action, U, ψ_wi, temp_force, smearing)

        dstaction_proj = real(multr(im * λ[group_direction], dSfdU_st[μ, site]))
        dstaction_proj_smeared = real(
            multr(im * λ[group_direction], dSfdU_st_smeared[μ, site])
        )
        dwiaction_proj = real(multr(im * λ[group_direction], dSfdU_wi[μ, site]))
        dwiaction_proj_smeared = real(
            multr(im * λ[group_direction], dSfdU_wi_smeared[μ, site])
        )

        st_symm_diff = (st_action_new_fwd - st_action_new_bwd) / 2ΔH
        st_symm_diff_smeared = (st_action_new_fwd_smeared - st_action_new_bwd_smeared) / 2ΔH
        wi_symm_diff = (wi_action_new_fwd - wi_action_new_bwd) / 2ΔH
        wi_symm_diff_smeared = (wi_action_new_fwd_smeared - wi_action_new_bwd_smeared) / 2ΔH

        relerrors[1, group_direction] = (st_symm_diff - dstaction_proj) / st_symm_diff
        relerrors[2, group_direction] =
            (st_symm_diff_smeared - dstaction_proj_smeared) / st_symm_diff_smeared
        relerrors[3, group_direction] = (wi_symm_diff - dwiaction_proj) / wi_symm_diff
        relerrors[4, group_direction] =
            (wi_symm_diff_smeared - dwiaction_proj_smeared) / wi_symm_diff_smeared

        # println("================= Group direction $(group_direction) =================")
        # println("/ Staggered rel. error (unsmeared): \t", relerrors[1, group_direction])
        # println("/ Staggered rel. error (smeared):   \t", relerrors[2, group_direction])
        # println("")
        # println(
        #     "/ Wilson rel. error (unsmeared): \t",
        #     (wi_symm_diff - dwiaction_proj) / wi_symm_diff,
        # )
        # println(
        #     "/ Wilson rel. error (smeared):   \t",
        #     (wi_symm_diff_smeared - dwiaction_proj_smeared) / wi_symm_diff_smeared,
        # )
    end
    println()
    return relerrors
end

SU3testfderivative()

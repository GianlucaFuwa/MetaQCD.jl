using MetaQCD
using MetaQCD.Utils
using Random

const StaggeredFermionAction = MetaQCD.DiracOperators.StaggeredFermionAction
const WilsonFermionAction = MetaQCD.DiracOperators.WilsonFermionAction

function SU3testfderivative(; dirac="staggered", backend=nothing)
    Random.seed!(123)
    println("Fermion derivative test [$dirac]")
    MetaQCD.Output.set_global_logger!(1, devnull; tc=false)
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    U = initial_gauges("cold", NX, NY, NZ, NT, 6.0; GA=WilsonGaugeAction)
    ψ = Fermionfield(NX, NY, NZ, NT; staggered=dirac=="staggered")
    MetaQCD.Gaugefields.gaussian_pseudofermions!(ψ)
    
    action = if dirac == "staggered" 
        StaggeredFermionAction(U, 1.0)
    elseif dirac == "wilson"
        WilsonFermionAction(U, 1.0)
    else
        error("dirac operator $dirac not supported")
    end

    filename = "./test/testconf.txt"
    loadU!(BridgeFormat(), U, filename);
    if backend !== nothing
        U = MetaQCD.to_backend(backend, U)
    end

    # Test for smearing with 5 steps and stout parameter 0.12
    smearing = StoutSmearing(U, 5, 0.12)

    dSfdU = Temporaryfield(U)
    dSfdU_smeared = Temporaryfield(U)
    temp_force = Temporaryfield(U)

    site = SiteCoords(2, 3, 1, 2)
    μ = 3
    ΔH = 0.000001

    relerrors = Matrix{Float64}(undef, 2, 8)

    for group_direction in 1:8
        # Unsmeared
        Ufwd = deepcopy(U)
        Ufwd[μ, site] = expλ(group_direction, ΔH) * Ufwd[μ, site]
        MetaQCD.DiracOperators.replace_U!(action.D, Ufwd)
        action_new_fwd = calc_fermion_action(action, ψ)

        Ubwd = deepcopy(U)
        Ubwd[μ, site] = expλ(group_direction, -ΔH) * Ubwd[μ, site]
        MetaQCD.DiracOperators.replace_U!(action.D, Ubwd)
        action_new_bwd = calc_fermion_action(action, ψ)

        # Smeared
        Ufwd = deepcopy(U)
        Ufwd[μ, site] = expλ(group_direction, ΔH) * Ufwd[μ, site]
        calc_smearedU!(smearing, Ufwd)
        MetaQCD.DiracOperators.replace_U!(action.D, smearing.Usmeared_multi[end])
        action_new_fwd_smeared = calc_fermion_action(action, ψ)

        Ubwd = deepcopy(U)
        Ubwd[μ, site] = expλ(group_direction, -ΔH) * Ubwd[μ, site]
        calc_smearedU!(smearing, Ubwd)
        MetaQCD.DiracOperators.replace_U!(action.D, smearing.Usmeared_multi[end])
        action_new_bwd_smeared = calc_fermion_action(action, ψ)

        calc_dSfdU_bare!(dSfdU, action, U, ψ, nothing, NoSmearing())
        calc_dSfdU_bare!(dSfdU_smeared, action, U, ψ, temp_force, smearing)

        daction_proj = real(multr(im * λ[group_direction], dSfdU[μ, site]))
        daction_proj_smeared = real(
            multr(im * λ[group_direction], dSfdU_smeared[μ, site])
        )

        symm_diff = (action_new_fwd - action_new_bwd) / 2ΔH
        symm_diff_smeared = (action_new_fwd_smeared - action_new_bwd_smeared) / 2ΔH

        relerrors[1, group_direction] = (symm_diff - daction_proj) / symm_diff
        relerrors[2, group_direction] =
            (symm_diff_smeared - daction_proj_smeared) / symm_diff_smeared

        # println("================= Group direction $(group_direction) =================")
        # println("/ Rel. error (unsmeared): \t", relerrors[1, group_direction])
        # println("/ Rel. error (smeared):   \t", relerrors[2, group_direction])
    end
    println()
    return relerrors
end

SU3testfderivative(; dirac="staggered")

using MetaQCD
using MetaQCD.Utils
using LinearAlgebra
using Random

const StaggeredFermionAction = MetaQCD.DiracOperators.StaggeredFermionAction
const WilsonFermionAction = MetaQCD.DiracOperators.WilsonFermionAction

function SU3testfderivative(; dirac="staggered", backend=nothing)
    Random.seed!(123)
    println("Fermion derivative test [$dirac]")
    MetaQCD.Output.set_global_logger!(1, devnull; tc=true)
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    U = initial_gauges("cold", NX, NY, NZ, NT, 6.0; GA=WilsonGaugeAction)
    ϕ = Fermionfield(NX, NY, NZ, NT; staggered=dirac == "staggered")
    ψ = Fermionfield(NX, NY, NZ, NT; staggered=dirac == "staggered")
    MetaQCD.Gaugefields.gaussian_pseudofermions!(ϕ)

    action = if dirac == "staggered"
        StaggeredFermionAction(U, 0.1; cg_tol=1e-16)
    elseif dirac == "wilson"
        WilsonFermionAction(U, 0.1; csw=1, cg_tol=1e-19, cg_maxiters=10000)
    else
        error("dirac operator $dirac not supported")
    end

    filename = "./test/testconf.txt"
    loadU!(BridgeFormat(), U, filename)
    if backend !== nothing
        U = MetaQCD.to_backend(backend, U)
    end
    mul!(ψ, action.D(U), ϕ)

    # Test for smearing with 5 steps and stout parameter 0.12
    smearing = StoutSmearing(U, 5, 0.12)

    dSfdU = Temporaryfield(U)
    dSfdU_smeared = Temporaryfield(U)
    temp_force = Temporaryfield(U)

    site = SiteCoords(2, 3, 1, 2)
    μ = 3
    ΔH = 0.000001

    relerrors = Matrix{Float64}(undef, 8, 2)

    for group_direction in 1:8
        # Unsmeared
        Ufwd = deepcopy(U)
        Ufwd[μ, site] = expλ(group_direction, ΔH) * Ufwd[μ, site]
        action_new_fwd = calc_fermion_action(action, Ufwd, ψ)

        Ubwd = deepcopy(U)
        Ubwd[μ, site] = expλ(group_direction, -ΔH) * Ubwd[μ, site]
        action_new_bwd = calc_fermion_action(action, Ubwd, ψ)

        # Smeared
        Ufwd = deepcopy(U)
        Ufwd[μ, site] = expλ(group_direction, ΔH) * Ufwd[μ, site]
        calc_smearedU!(smearing, Ufwd)
        action_new_fwd_smeared = calc_fermion_action(
            action, smearing.Usmeared_multi[end], ψ
        )

        Ubwd = deepcopy(U)
        Ubwd[μ, site] = expλ(group_direction, -ΔH) * Ubwd[μ, site]
        calc_smearedU!(smearing, Ubwd)
        action_new_bwd_smeared = calc_fermion_action(
            action, smearing.Usmeared_multi[end], ψ
        )

        calc_dSfdU_bare!(dSfdU, action, U, ψ, nothing, NoSmearing())
        calc_dSfdU_bare!(dSfdU_smeared, action, U, ψ, temp_force, smearing)

        daction_proj = real(multr(im * λ[group_direction], dSfdU[μ, site]))
        daction_proj_smeared = real(multr(im * λ[group_direction], dSfdU_smeared[μ, site]))

        symm_diff = (action_new_fwd - action_new_bwd) / 2ΔH
        symm_diff_smeared = (action_new_fwd_smeared - action_new_bwd_smeared) / 2ΔH

        # @show daction_proj
        # @show symm_diff
        relerrors[group_direction, 1] = (symm_diff - daction_proj) / symm_diff
        relerrors[group_direction, 2] =
            (symm_diff_smeared - daction_proj_smeared) / symm_diff_smeared

        # println("================= Group direction $(group_direction) =================")
        # println("/ Rel. error (unsmeared): \t", relerrors[1, group_direction])
        # println("/ Rel. error (smeared):   \t", relerrors[2, group_direction])
    end
    println()
    return relerrors
end

SU3testfderivative(; dirac="staggered")

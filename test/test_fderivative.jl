using MetaQCD
using MetaQCD
using MetaQCD.Utils
using LinearAlgebra
using Random

function test_fderivative(
    backend=CPU; dirac="staggered", mass=0.01, eoprec=false, single_flavor=false
)
    Random.seed!(123)
    println("Fermion derivative test [$dirac]")
    MetaQCD.Output.set_global_logger!(1, nothing; tc=true)
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    U = Gaugefield{CPU,Float64,WilsonGaugeAction}(NX, NY, NZ, NT, 6.0)
    random_gauges!(U)

    # filename = pkgdir(MetaQCD, "test", "testconf.txt")
    # load_config!(BridgeFormat(), U, filename)
    if backend !== CPU
        U = MetaQCD.to_backend(backend, U)
    end

    ψ = eoprec ? even_odd(Fermionfield(U; staggered=dirac=="staggered")) : Fermionfield(U; staggered=dirac=="staggered")

    action = if dirac == "staggered"
        if eoprec
            Nf = single_flavor ? 1 : 4
            StaggeredEOPreFermionAction(
                U,
                mass;
                Nf=Nf,
                cg_maxiters_action=5000,
                cg_maxiters_md=5000,
                cg_tol_action=1e-16,
                cg_tol_md=1e-16,
                rhmc_prec_action=64
            )
        else
            Nf = single_flavor ? 1 : 8
            StaggeredFermionAction(
                U,
                mass;
                Nf=Nf,
                cg_maxiters_action=5000,
                cg_maxiters_md=5000,
                cg_tol_action=1e-16,
                cg_tol_md=1e-16,
                rhmc_prec_action=64
            )
        end
    elseif dirac == "wilson"
        if eoprec
            Nf = single_flavor ? 1 : 2
            @assert Nf == 2
            WilsonEOPreFermionAction(
                U,
                mass;
                Nf=Nf,
                csw=1.78,
                cg_maxiters_action=5000,
                cg_maxiters_md=5000,
                cg_tol_action=1e-16,
                cg_tol_md=1e-16,
                rhmc_prec_action=64
            )
        else
            Nf = single_flavor ? 1 : 2
            WilsonFermionAction(
                U,
                mass;
                Nf=Nf,
                csw=1.78,
                cg_maxiters_action=5000,
                cg_maxiters_md=5000,
                cg_tol_action=1e-16,
                cg_tol_md=1e-16,
                rhmc_prec_action=64
            )
        end
    else
        error("dirac operator $dirac not supported")
    end
    @show action

    sample_pseudofermions!(ψ, action, U)

    # Test for smearing with 5 steps and stout parameter 0.12
    smearing = StoutSmearing(U, 5, 0.12)

    dSfdU = Colorfield(U)
    dSfdU_smeared = Colorfield(U)
    temp_force = Colorfield(U)

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

        # if group_direction == 1
        #     @show daction_proj
        #     @show symm_diff
        # end
        relerrors[group_direction, 1] = (symm_diff - daction_proj) / symm_diff
        relerrors[group_direction, 2] =
            (symm_diff_smeared - daction_proj_smeared) / symm_diff_smeared

        println("================= Group direction $(group_direction) =================")
        println("/ Rel. error (unsmeared): \t", relerrors[group_direction, 1])
        println("/ Rel. error (smeared):   \t", relerrors[group_direction, 2])
    end
    println()
    return relerrors
end

using MetaQCD
using MetaQCD.Utils
using MetaQCD.Fields: update_halo!
using LinearAlgebra
using Random
using ProfileView

function test_mpi_fderivative(
    backend=CPU; dirac="staggered", mass=0.01, eoprec=false, single_flavor=false, csw=1.78
)
    if mpi_amroot()
        println("Fermion derivative test [$dirac]")
    end

    MetaQCD.MetaIO.set_global_logger!(1, nothing; tc=true)
    Ns = 4
    Nt = 4
    U = Gaugefield{CPU,Float64,WilsonGaugeAction}(Ns, Ns, Ns, Nt, 5.7, (1, 1, 2, 2), 1)
    random_gauges!(U)

    # filename = pkgdir(MetaQCD, "test", "testconf.txt")
    # load_config!(BridgeFormat(), U, filename)
    if backend !== CPU
        U = MetaQCD.to_backend(backend, U)
    end

    ψ = eoprec ? even_odd(Spinorfield(U; staggered=dirac=="staggered")) : Spinorfield(U; staggered=dirac=="staggered")

    spectral_bound, Nf = if dirac=="staggered"
        (mass^2, 6.0), (single_flavor ? 1 : (eoprec ? 4 : 8))
    elseif dirac=="wilson"
        (mass^2, 64.0), (single_flavor ? 1 : 2)
    end

    params = (
        fermion_action=dirac,
        eo_precon=eoprec,
        boundary_condition="antiperiodic",
        rhmc_spectral_bound=spectral_bound,
        rhmc_order_md=15,
        rhmc_prec_md=64,
        rhmc_order_action=15,
        rhmc_prec_action=64,
        cg_tol_action=1e-16,
        cg_tol_md=1e-16,
        cg_maxiters_action=5000,
        cg_maxiters_md=5000,
        wilson_r=1,
        wilson_csw=csw,
    )

    action = MetaQCD.DiracOperators.init_fermion_action(params, mass, Nf, U)
    mpi_amroot() && (@show action)

    sample_pseudofermions!(ψ, action, U)

    # Test for smearing with 5 steps and stout parameter 0.12
    smearing = StoutSmearing(U, 5, 0.12)

    dSfdU = Colorfield(U)
    dSfdU_smeared = Colorfield(U)
    temp_force = Colorfield(U)

    site = SiteCoords(2, 3, 2, 2)
    μ = 3
    ΔH = 0.000001

    relerrors = Matrix{Float64}(undef, 8, 2)

    for group_direction in 1:8
        # Unsmeared
        Ufwd = deepcopy(U)
        if mpi_amroot()
            Ufwd[μ, site] = expλ(group_direction, ΔH) * Ufwd[μ, site]
        end
        update_halo!(Ufwd)
        action_new_fwd = calc_fermion_action(action, Ufwd, ψ)

        Ubwd = deepcopy(U)
        if mpi_amroot()
            Ubwd[μ, site] = expλ(group_direction, -ΔH) * Ubwd[μ, site]
        end
        update_halo!(Ubwd)
        action_new_bwd = calc_fermion_action(action, Ubwd, ψ)

        # Smeared
        Ufwd = deepcopy(U)
        if mpi_amroot()
            Ufwd[μ, site] = expλ(group_direction, ΔH) * Ufwd[μ, site]
        end
        update_halo!(Ufwd)
        calc_smearedU!(smearing, Ufwd)
        action_new_fwd_smeared = calc_fermion_action(
            action, smearing.Usmeared_multi[end], ψ
        )

        Ubwd = deepcopy(U)
        if mpi_amroot()
            Ubwd[μ, site] = expλ(group_direction, -ΔH) * Ubwd[μ, site]
        end
        update_halo!(Ubwd)
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

        if mpi_amroot()
            println("================= Group direction $(group_direction) =================")
            println("/ Rel. error (unsmeared): \t", relerrors[group_direction, 1])
            println("/ Rel. error (smeared):   \t", relerrors[group_direction, 2])
        end
    end
    # println()
    return relerrors
end

test_mpi_fderivative()

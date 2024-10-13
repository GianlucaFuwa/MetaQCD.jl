using MetaQCD
using MetaQCD.Utils
using LinearAlgebra
using Random

function test_fderivative(
    backend=CPU;
    nprocs_cart=(1, 1, 1, 1),
    halo_width=0,
    dirac="staggered",
    mass=0.01,
    eoprec=false,
    single_flavor=false,
    csw=1.78
)
    if mpi_amroot()
        println("Fermion derivative test [$dirac]")
    end

    Random.seed!(123)
    MetaQCD.MetaIO.set_global_logger!(1, nothing; tc=true)
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    U = Gaugefield{CPU,Float64,WilsonGaugeAction}(
        NX, NY, NZ, NT, 6.0, nprocs_cart, halo_width
    )
    random_gauges!(U)

    filename = pkgdir(MetaQCD, "test", "testconf.txt")
    load_config!(BridgeFormat(), U, filename)
    if backend !== CPU
        U = MetaQCD.to_backend(backend, U)
    end

    ψ = if eoprec
        even_odd(Spinorfield(U; staggered=dirac=="staggered"))
    else
        Spinorfield(U; staggered=dirac=="staggered")
    end

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
    # mpi_amroot() && (@show action)

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

    if mpi_amroot()
        println()
        # @test length(findall(x -> abs(x) > 1e-4, relerrors[:, 2])) == 0
    end

    mpi_barrier()
    return relerrors
end

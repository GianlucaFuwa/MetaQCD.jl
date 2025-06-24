using MetaQCD
using MetaQCD.Utils
using LinearAlgebra
using Random
using Test

function test_fderivative(;
    backend=CPU,
    nprocs_cart=(1, 1, 1, 1),
    halo_width=1,
    dirac="staggered",
    eoprec=false,
    mass=0.01,
    single_flavor=false,
    csw=1.78
)
    if mpi_amroot()
        println("Fermion derivative test [$dirac]")
    end

    relerrors = Matrix{Float64}(undef, 8, 2)
    csw_str = if dirac == "wilson"
        csw == 0 ? " (no clover)" : " (clover)"
    else
        ""
    end
    name_str ="$dirac$(ifelse(eoprec, " even-odd", ""))"
    MetaQCD.MetaIO.set_global_logger!(4, nothing; tc=true)

    @testset "$(name_str)$(csw_str) derivative" begin
        Random.seed!(123 * (mpi_myrank() + 1))
        NX = 4
        NY = 4
        NZ = 4
        NT = 4
        U = Gaugefield{backend,Float64,WilsonGaugeAction}(
            NX, NY, NZ, NT, 6.0, numprocs_cart=nprocs_cart, halo_width=halo_width
        )
        filename = if nprocs_cart != (1, 1, 1, 1)
            pkgdir(MetaQCD, "test", "testconf_mpi")
        else
            pkgdir(MetaQCD, "test", "testconf.txt")
        end
        if backend==CPU
            load_config!(BridgeFormat(), U, filename)
        else
            random_gauges!(U)
        end

        # if backend !== CPU
        #     U = MetaQCD.to_backend(backend, U)
        # end

        is_staggered = contains(dirac, "staggered")
        is_hoelbling = dirac ∈ ("staggered-h1234", "staggered-h1324", "staggered-h1342")

        ψ = if eoprec
            even_odd(Spinorfield(U; staggered=is_staggered))
        else
            Spinorfield(U; staggered=is_staggered)
        end

        spectral_bound, Nf = if is_staggered && !is_hoelbling
            (mass^2, 6.0), (single_flavor ? 1 : (eoprec ? 4 : 8))
        else
            (mass^2, 64.0), (single_flavor ? 1 : 2)
        end

        action = MetaQCD.DiracOperators.FermionAction(
            dirac*ifelse(eoprec, "_eo", ""),
            U,
            mass,
            bc_str="antiperiodic",
            rhmc_spectral_bound=spectral_bound,
            rhmc_tol_md = 0.01,
            rhmc_tol_action = 0.01,
            rhmc_order_md=12,
            rhmc_order_action=12,
            cg_tol_action=1e-8,
            cg_tol_md=1e-8,
            cg_maxiters_action=1000,
            cg_maxiters_md=1000,
            csw=csw,
        )
        mpi_amroot() && (@show action)

        sample_pseudofermions!(ψ, action, U)

        # Test for smearing with 5 steps and stout parameter 0.12
        smearing = StoutSmearing(U; numlayers=5, rho=0.12)

        dSfdU = Colorfield(U)
        dSfdU_smeared = Colorfield(U)
        temp_force = Colorfield(U)

        coord = (2, 3, 1, 2)
        site = SiteCoords(coord...)
        μ = 3
        ΔH = 0.000001

        for group_direction in 1:8
            # Unsmeared
            Ufwd = deepcopy(U)
            if site in eachindex(U)
                Ufwd[μ, site] = expλ(group_direction, ΔH) * Ufwd[μ, site]
            end
            action_new_fwd = calc_fermion_action(action, Ufwd, ψ)

            Ubwd = deepcopy(U)
            if site in eachindex(U)
                Ubwd[μ, site] = expλ(group_direction, -ΔH) * Ubwd[μ, site]
            end
            action_new_bwd = calc_fermion_action(action, Ubwd, ψ)

            # Smeared
            Ufwd = deepcopy(U)
            if site in eachindex(U)
                Ufwd[μ, site] = expλ(group_direction, ΔH) * Ufwd[μ, site]
            end
            calc_smearedU!(smearing, Ufwd)
            action_new_fwd_smeared = calc_fermion_action(
                action, smearing.Usmeared_multi[end], ψ
            )

            Ubwd = deepcopy(U)
            if site in eachindex(U)
                Ubwd[μ, site] = expλ(group_direction, -ΔH) * Ubwd[μ, site]
            end
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

            if group_direction == 1
                @show daction_proj
                @show symm_diff
            end
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
            @test sum(relerrors[:, 2]) / length(relerrors[:, 2]) < 1e-1
        end
    end

    mpi_barrier()
    return relerrors
end

# test_fderivative(nprocs_cart=(1, 1, 2, 1), single_flavor=true, halo_width=1)

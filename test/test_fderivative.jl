using MetaQCD
using MetaQCD.Utils
using LinearAlgebra
using Random
using Test
# using AMDGPU

function test_fderivative(;
    backend=CPU,
    nprocs_cart=(1, 1, 1, 1),
    halo_width=1,
    N=4,
    dirac="staggered",
    eoprec=false,
    mass=0.01,
    single_flavor=false,
    csw=1.78,
    do_test=true,
)
    if mpi_amroot()
        println("Fermion derivative test [$dirac]")
    end
    @assert N in (4, 16)

    relerrors = Matrix{Float64}(undef, 8, 2)
    csw_str = if dirac == "wilson"
        csw == 0 ? " (no clover)" : " (clover)"
    else
        ""
    end
    name_str ="$dirac$(ifelse(eoprec, " even-odd", ""))"
    MetaQCD.MetaIO.set_global_logger!(1, nothing; tc=true)

    @testset "$(name_str)$(csw_str) derivative" begin
        Random.seed!(123 * (mpi_myrank() + 1))
        NX = NY = NZ = NT = N
        Ucpu = Gaugefield{CPU,Float64,WilsonGaugeAction,12}(
            NX, NY, NZ, NT, 6.0, numprocs_cart=nprocs_cart, halo_width=halo_width
        )
        filename = if nprocs_cart != (1, 1, 1, 1)
            pkgdir(MetaQCD, "test", NX==4 ? "testconf_mpi" : "testconf_16_mpi")
        else
            pkgdir(MetaQCD, "test", NX==4 ? "testconf.txt" : "testconf_16.txt")
        end

        load_field!(BridgeFormat(), Ucpu, filename)

        if backend !== CPU
            U = MetaQCD.convert_field(backend, Ucpu)
        else
            U = Ucpu
        end

        is_staggered = contains(dirac, "staggered")
        is_hoelbling = dirac ∈ ("staggered-h1234", "staggered-h1324", "staggered-h1342")

        rhmc_spectral_bound, Nf = if is_staggered && !is_hoelbling
            (mass^2, 6.0), (single_flavor ? 1 : (eoprec ? 4 : 8))
        else
            (mass^2, 64.0), (single_flavor ? 1 : 2)
        end

        action = MetaQCD.DiracOperators.FermionAction(
            dirac*ifelse(eoprec, "_eo", ""),
            U,
            mass;
            bc_str="antiperiodic",
            Nf,
            rhmc_spectral_bound,
            rhmc_tol_md=0.1,
            rhmc_tol_action=0.1,
            rhmc_order_md=15,
            rhmc_order_action=15,
            cg_tol_action=1e-8,
            cg_tol_md=1e-8,
            cg_maxiters_action=5000,
            cg_maxiters_md=5000,
            csw=csw,
        )
        mpi_amroot() && (@show action)

        ψ = if eoprec
            even_odd(Spinorfield(action.D.temp; staggered=is_staggered))
        else
            Spinorfield(action.D.temp; staggered=is_staggered)
        end

        # mpi_amroot() && println("sample pseudofermions")
        sample_pseudofermions!(ψ, action, U)

        # Test for smearing with 5 steps and stout parameter 0.12
        # mpi_amroot() && println("smearing")
        smearing = StoutSmearing(U; numlayers=5, rho=0.12)

        # mpi_amroot() && println("temps")
        dSfdU = Colorfield(U)
        dSfdU_smeared = Colorfield(U)
        temp_force = Colorfield(U)

        coord = (2, 3, 1, 1)
        site = SiteCoords(coord...)
        μ = 3
        ΔH = 0.000001

        for group_direction in 1:8
            # Unsmeared
            # mpi_amroot() && println("$(group_direction) unsmeared fwd")
            Ufwdcpu = deepcopy(Ucpu)
            if site in eachindex(Ucpu)
                Ufwdcpu[μ, site] = expλ(group_direction, ΔH) * Ufwdcpu[μ, site]
            end
            Ufwd = convert_field(backend, Ufwdcpu)
            action_new_fwd = calc_fermion_action(action, Ufwd, ψ)

            # mpi_amroot() && println("$(group_direction) unsmeared bwd")
            Ubwdcpu = deepcopy(Ucpu)
            if site in eachindex(Ucpu)
                Ubwdcpu[μ, site] = expλ(group_direction, -ΔH) * Ubwdcpu[μ, site]
            end
            Ubwd = convert_field(backend, Ubwdcpu)
            action_new_bwd = calc_fermion_action(action, Ubwd, ψ)

            # Smeared
            # mpi_amroot() && println("$(group_direction) smeared fwd")
            calc_smearedU!(smearing, Ufwd)
            action_new_fwd_smeared = calc_fermion_action(
                action, smearing.Usmeared_multi[end], ψ
            )

            # mpi_amroot() && println("$(group_direction) smeared bwd")
            calc_smearedU!(smearing, Ubwd)
            action_new_bwd_smeared = calc_fermion_action(
                action, smearing.Usmeared_multi[end], ψ
            )

            calc_dSfdU_bare!(dSfdU, action, U, ψ, nothing, NoSmearing())
            calc_dSfdU_bare!(dSfdU_smeared, action, U, ψ, temp_force, smearing)

            if site in eachindex(U)
                daction_proj = real(multr(im * λ[group_direction], dSfdU[μ, site]))
                daction_proj_smeared = real(multr(im * λ[group_direction], dSfdU_smeared[μ, site]))
            else
                daction_proj = 1.0
                daction_proj_smeared = 1.0
            end

            symm_diff = (action_new_fwd - action_new_bwd) / 2ΔH
            symm_diff_smeared = (action_new_fwd_smeared - action_new_bwd_smeared) / 2ΔH

            relerrors[group_direction, 1] = (symm_diff - daction_proj) / symm_diff
            relerrors[group_direction, 2] =
                (symm_diff_smeared - daction_proj_smeared) / symm_diff_smeared

            if group_direction == 1
                @show daction_proj
                @show symm_diff
            end

            if mpi_amroot()
                println("================= Group direction $(group_direction) =================")
                println("/ Rel. error (unsmeared): \t", relerrors[group_direction, 1])
                println("/ Rel. error (smeared):   \t", relerrors[group_direction, 2])
            end
        end

        if mpi_amroot() && do_test
            println()
            @test sum(relerrors[:, 2]) / length(relerrors[:, 2]) < 1e-1
        end
    end

    mpi_barrier()
    return relerrors
end

# AMDGPU.@allowscalar test_fderivative(;
#     single_flavor=true, backend=ROCBackend, nprocs_cart=(1, 1, 1, mpi_size())
# )
test_fderivative(; dirac="wilson", nprocs_cart=(1, 1, 1, mpi_size()), csw=1.0)

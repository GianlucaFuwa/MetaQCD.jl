using Random
using MetaQCD
using MetaQCD.Utils
using MetaQCD.Measurements: top_charge_deriv!
using Test

function test_derivative(; backend=CPU, GA=WilsonGaugeAction, nprocs_cart=(1, 1, 1, 1), halo_width=1)
    Random.seed!(123)
    mpi_amroot() && println("Gauge and Clover derivative test")

    relerrors = Matrix{Float64}(undef, 8, 4)

    @testset "Gauge derivative" begin
        NX = NY = NZ = NT = 4
        Ucpu = Gaugefield{CPU,Float64,GA,12}(
            NX, NY, NZ, NT, 6.0, numprocs_cart=nprocs_cart, halo_width=halo_width
        )
        filename = if nprocs_cart != (1, 1, 1, 1)
            pkgdir(MetaQCD, "test", "testconf_mpi")
        else
            pkgdir(MetaQCD, "test", "testconf.txt")
        end

        load_field!(BridgeFormat(), Ucpu, filename)

        if backend !== CPU
            U = MetaQCD.convert_field(backend, Ucpu)
        else
            U = Ucpu
        end

        # gaction_old = calc_gauge_action(U)
        # topcharge_old = top_charge(U)

        # Test for smearing with 5 steps and stout parameter 0.12
        smearing = StoutSmearing(U; numlayers=5, rho=0.12)

        staples = Colorfield(U)
        fieldstrength = Tensorfield(U)
        temp_force = Colorfield(U)
        dSdU = Colorfield(U)
        dSdU_smeared = Colorfield(U)
        dQdU = Colorfield(U)
        dQdU_smeared = Colorfield(U)

        coord = (2, 3, 1, 2)
        site = SiteCoords(coord...)
        μ = 3
        ΔH = 0.00001

        for group_direction in 1:8
            # Unsmeared
            Ufwdcpu = deepcopy(Ucpu)
            if site in eachindex(Ucpu)
                Ufwdcpu[μ, site] = expλ(group_direction, ΔH) * Ufwdcpu[μ, site]
            end
            Ufwd = convert_field(backend, Ufwdcpu)
            gaction_new_fwd = calc_gauge_action(Ufwd)
            topcharge_new_fwd = top_charge(Clover(), Ufwd)

            Ubwdcpu = deepcopy(Ucpu)
            if site in eachindex(Ucpu)
                Ubwdcpu[μ, site] = expλ(group_direction, -ΔH) * Ubwdcpu[μ, site]
            end
            Ubwd = convert_field(backend, Ubwdcpu)
            gaction_new_bwd = calc_gauge_action(Ubwd)
            topcharge_new_bwd = top_charge(Clover(), Ubwd)

            # Smeared
            calc_smearedU!(smearing, Ufwd)
            gaction_new_fwd_smeared = calc_gauge_action(smearing.Usmeared_multi[end])
            topcharge_new_fwd_smeared = top_charge(Clover(), smearing.Usmeared_multi[end])

            calc_smearedU!(smearing, Ubwd)
            gaction_new_bwd_smeared = calc_gauge_action(smearing.Usmeared_multi[end])
            topcharge_new_bwd_smeared = top_charge(Clover(), smearing.Usmeared_multi[end])

            calc_dSdU_bare!(dSdU, staples, U, nothing, NoSmearing())
            calc_dSdU_bare!(dSdU_smeared, staples, U, temp_force, smearing)
            top_charge_deriv_bare!(Clover(), dQdU, fieldstrength, U, nothing, NoSmearing())
            top_charge_deriv_bare!(Clover(), dQdU_smeared, fieldstrength, U, temp_force, smearing)

            dgaction_proj = real(multr(im * λ[group_direction], dSdU[μ, site]))
            dtopcharge_proj = real(multr(im * λ[group_direction], dQdU[μ, site]))
            dgaction_proj_smeared = real(multr(im * λ[group_direction], dSdU_smeared[μ, site]))
            dtopcharge_proj_smeared = real(
                multr(im * λ[group_direction], dQdU_smeared[μ, site])
            )

            ga_symm_diff = (gaction_new_fwd - gaction_new_bwd) / 2ΔH
            tc_symm_diff = (topcharge_new_fwd - topcharge_new_bwd) / 2ΔH
            ga_symm_diff_smeared = (gaction_new_fwd_smeared - gaction_new_bwd_smeared) / 2ΔH
            tc_symm_diff_smeared = (topcharge_new_fwd_smeared - topcharge_new_bwd_smeared) / 2ΔH

            relerrors[group_direction, 1] = (ga_symm_diff - dgaction_proj) / ga_symm_diff
            relerrors[group_direction, 2] =
                (ga_symm_diff_smeared - dgaction_proj_smeared) / ga_symm_diff_smeared
            relerrors[group_direction, 3] = (tc_symm_diff - dtopcharge_proj) / tc_symm_diff
            relerrors[group_direction, 4] =
                (tc_symm_diff_smeared - dtopcharge_proj_smeared) / tc_symm_diff_smeared

            if mpi_amroot()
                println("================= Group direction $(group_direction) =================")
                println(
                    "/ GA rel. error (unsmeared): \t", (ga_symm_diff - dgaction_proj) / ga_symm_diff
                )
                println(
                    "/ GA rel. error (smeared):   \t",
                    (ga_symm_diff_smeared - dgaction_proj_smeared) / ga_symm_diff_smeared,
                )
                println("")
                println(
                    "/ TC rel. error (unsmeared): \t",
                    (tc_symm_diff - dtopcharge_proj) / tc_symm_diff,
                )
                println(
                    "/ TC rel. error (smeared):   \t",
                    (tc_symm_diff_smeared - dtopcharge_proj_smeared) / tc_symm_diff_smeared,
                )
            end
        end

        if mpi_amroot()
            println()

            @testset "Gauge Action derivative" begin
                @test sum(relerrors[:, 2]) / length(relerrors[:, 2]) < 1e-4
            end

            @testset "Clover derivative" begin
                @test sum(relerrors[:, 4]) / length(relerrors[:, 4]) < 1e-4
            end
        end
    end

    mpi_barrier()
    return relerrors
end

function top_charge_deriv_bare!(kind_of_charge, dU, F, U, temp_force, smearing)
    if isnothing(temp_force)
        top_charge_deriv!(dU, F, U, kind_of_charge)
    else
        calc_smearedU!(smearing, U)
        fully_smeared_U = smearing.Usmeared_multi[end]
        top_charge_deriv!(dU, F, fully_smeared_U, kind_of_charge)
        stout_backprop!(dU, temp_force, smearing)
    end

    return nothing
end

test_derivative(; nprocs_cart=(1, 1, 1, 2))

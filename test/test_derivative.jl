using Random
using MetaQCD
using MetaQCD.Utils
using MetaQCD.Updates: calc_dQdU_bare!

function test_derivative(backend=CPU)
    Random.seed!(123)
    println("SU3test_gauge_derivative")
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    U = Gaugefield{backend,Float64,WilsonGaugeAction}(NX, NY, NZ, NT, 6.0)
    random_gauges!(U)
    # filename = "./test/testconf.txt"
    # loadU!(BridgeFormat(), U, filename);
    if backend !== nothing
        U = MetaQCD.to_backend(backend, U)
    end

    # gaction_old = calc_gauge_action(U)
    # topcharge_old = top_charge(Clover(), U)

    # Test for smearing with 5 steps and stout parameter 0.12
    smearing = StoutSmearing(U, 5, 0.12)

    staples = Colorfield(U)
    fieldstrength = Tensorfield(U)
    temp_force = Colorfield(U)
    dSdU = Colorfield(U)
    dSdU_smeared = Colorfield(U)
    dQdU = Colorfield(U)
    dQdU_smeared = Colorfield(U)

    site = SiteCoords(2, 3, 1, 2)
    μ = 3
    ΔH = 0.00001

    relerrors = Matrix{Float64}(undef, 8, 4)

    for group_direction in 1:8
        # Unsmeared
        Ufwd = deepcopy(U)
        Ufwd[μ, site] = expλ(group_direction, ΔH) * Ufwd[μ, site]
        gaction_new_fwd = calc_gauge_action(Ufwd)
        topcharge_new_fwd = top_charge(Clover(), Ufwd)

        Ubwd = deepcopy(U)
        Ubwd[μ, site] = expλ(group_direction, -ΔH) * Ubwd[μ, site]
        gaction_new_bwd = calc_gauge_action(Ubwd)
        topcharge_new_bwd = top_charge(Clover(), Ubwd)

        # Smeared
        Ufwd = deepcopy(U)
        Ufwd[μ, site] = expλ(group_direction, ΔH) * Ufwd[μ, site]
        calc_smearedU!(smearing, Ufwd)
        gaction_new_fwd_smeared = calc_gauge_action(smearing.Usmeared_multi[end])
        topcharge_new_fwd_smeared = top_charge(Clover(), smearing.Usmeared_multi[end])

        Ubwd = deepcopy(U)
        Ubwd[μ, site] = expλ(group_direction, -ΔH) * Ubwd[μ, site]
        calc_smearedU!(smearing, Ubwd)
        gaction_new_bwd_smeared = calc_gauge_action(smearing.Usmeared_multi[end])
        topcharge_new_bwd_smeared = top_charge(Clover(), smearing.Usmeared_multi[end])

        calc_dSdU_bare!(dSdU, staples, U, nothing, NoSmearing())
        calc_dSdU_bare!(dSdU_smeared, staples, U, temp_force, smearing)
        calc_dQdU_bare!(Clover(), dQdU, fieldstrength, U, nothing, NoSmearing())
        calc_dQdU_bare!(Clover(), dQdU_smeared, fieldstrength, U, temp_force, smearing)

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

        # @show tc_symm_diff
        # @show dtopcharge_proj
        # @show tc_symm_diff_smeared
        # @show dtopcharge_proj_smeared
        # println("---")
        relerrors[group_direction, 1] = (ga_symm_diff - dgaction_proj) / ga_symm_diff
        relerrors[group_direction, 2] =
            (ga_symm_diff_smeared - dgaction_proj_smeared) / ga_symm_diff_smeared
        relerrors[group_direction, 3] = (tc_symm_diff - dtopcharge_proj) / tc_symm_diff
        relerrors[group_direction, 4] =
            (tc_symm_diff_smeared - dtopcharge_proj_smeared) / tc_symm_diff_smeared

    #     println("================= Group direction $(group_direction) =================")
    #     println(
    #         "/ GA rel. error (unsmeared): \t", (ga_symm_diff - dgaction_proj) / ga_symm_diff
    #     )
    #     println(
    #         "/ GA rel. error (smeared):   \t",
    #         (ga_symm_diff_smeared - dgaction_proj_smeared) / ga_symm_diff_smeared,
    #     )
    #     println("")
    #     println(
    #         "/ TC rel. error (unsmeared): \t",
    #         (tc_symm_diff - dtopcharge_proj) / tc_symm_diff,
    #     )
    #     println(
    #         "/ TC rel. error (smeared):   \t",
    #         (tc_symm_diff_smeared - dtopcharge_proj_smeared) / tc_symm_diff_smeared,
    #     )
    end
    # println()
    return relerrors
end

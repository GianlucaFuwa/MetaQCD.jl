using MetaQCD.Updates: calc_dQdU_bare!

function SU3testderivative(backend=nothing)
    Random.seed!(123)
    println("SU3test_gauge_derivative")
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    U = initial_gauges("hot", NX, NY, NZ, NT, 6.0; GA=WilsonGaugeAction)
    # filename = "./test/testconf.txt"
    # loadU!(BridgeFormat(), U, filename);
    if backend !== nothing
        U = MetaQCD.to_backend(backend, U)
    end

    # gaction_old = calc_gauge_action(U)
    # topcharge_old = top_charge(Clover(), U)

    # Test for smearing with 5 steps and stout parameter 0.12
    smearing = StoutSmearing(U, 5, 0.12)

    staples = Temporaryfield(U)
    fieldstrength = Tensorfield(U)
    temp_force = Temporaryfield(U)
    dSdU = Temporaryfield(U)
    dSdU_smeared = Temporaryfield(U)
    dQdU = Temporaryfield(U)
    dQdU_smeared = Temporaryfield(U)

    site = SiteCoords(2, 3, 1, 2)
    μ = 3
    ΔH = 0.00001

    relerrors = Matrix{Float64}(undef, 4, 8)

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
        calc_dQdU_bare!(Clover(), dQdU, fieldstrength, U)
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

        relerrors[1, group_direction] = (ga_symm_diff - dgaction_proj) / ga_symm_diff
        relerrors[2, group_direction] =
            (ga_symm_diff_smeared - dgaction_proj_smeared) / ga_symm_diff_smeared
        relerrors[3, group_direction] = (tc_symm_diff - dtopcharge_proj) / tc_symm_diff
        relerrors[4, group_direction] =
            (tc_symm_diff_smeared - dtopcharge_proj_smeared) / tc_symm_diff_smeared

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
    println()
    return relerrors
end

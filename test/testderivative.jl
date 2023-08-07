function SU3testderivative()
    Random.seed!(1206)
    println("SU3testderivative")
    NX = 8; NY = 8; NZ = 8; NT = 8;
    U = random_gauges(NX, NY, NZ, NT, 5.7, type_of_gaction = DBW2GaugeAction)

    gaction_old = calc_gauge_action(U)
    topcharge_old = top_charge(U, "clover")

    # Test for smearing with 5 steps and stout parameter 0.12
    smearing = StoutSmearing(U, 5, 0.12)

    staples = Temporaryfield(U);
    fieldstrength = Vector{Temporaryfield}(undef, 4)

    for i in 1:4
        fieldstrength[i] = Temporaryfield(U)
    end

    temp_force = Temporaryfield(U)
    dSdU = Temporaryfield(U)
    dSdU_smeared = Temporaryfield(U)
    dQdU = Temporaryfield(U)
    dQdU_smeared = Temporaryfield(U)

    site = SiteCoords(2, 3, 1, 2)
    direction = 3
    deltaH = 0.00001

    relerrors = Matrix{Float64}(undef, 4, 8)

    for group_direction in 1:8
        # Unsmeared
        Ufwd = deepcopy(U)
        Ufwd[direction][site] = expλ(group_direction, deltaH) * Ufwd[direction][site]
        gaction_new_fwd = calc_gauge_action(Ufwd)
        topcharge_new_fwd = top_charge(Ufwd, "clover")

        Ubwd = deepcopy(U)
        Ubwd[direction][site] = expλ(group_direction, -deltaH) * Ubwd[direction][site]
        gaction_new_bwd = calc_gauge_action(Ubwd)
        topcharge_new_bwd = top_charge(Ubwd, "clover")

        # Smeared
        Ufwd = deepcopy(U)
        Ufwd[direction][site] = expλ(group_direction, deltaH) * Ufwd[direction][site]
        calc_smearedU!(smearing, Ufwd)
        gaction_new_fwd_smeared = calc_gauge_action(smearing.Usmeared_multi[end])
        topcharge_new_fwd_smeared = top_charge(smearing.Usmeared_multi[end], "clover")

        Ubwd = deepcopy(U)
        Ubwd[direction][site] = expλ(group_direction, -deltaH) * Ubwd[direction][site]
        calc_smearedU!(smearing, Ubwd)
        gaction_new_bwd_smeared = calc_gauge_action(smearing.Usmeared_multi[end])
        topcharge_new_bwd_smeared = top_charge(smearing.Usmeared_multi[end], "clover")

        MetaQCD.Updates.calc_dSdU!(
            dSdU,
            staples,
            U,
        )
        MetaQCD.Updates.calc_dSdU_bare!(
            dSdU_smeared,
            temp_force,
            staples,
            U,
            smearing,
        )
        MetaQCD.Updates.calc_dQdU!(
            dQdU,
            fieldstrength,
            U,
            "clover",
        )
        MetaQCD.Updates.calc_dQdU_bare!(
            dQdU_smeared,
            temp_force,
            fieldstrength,
            U,
            "clover",
            smearing,
        )


        dgaction_proj = real(multr(im * λ(group_direction), dSdU[direction][site]))
        dtopcharge_proj = real(multr(im * λ(group_direction), dQdU[direction][site]))
        dgaction_proj_smeared = real(multr(im * λ(group_direction), dSdU_smeared[direction][site]))
        dtopcharge_proj_smeared = real(multr(im * λ(group_direction), dQdU_smeared[direction][site]))

        ga_symm_diff = (gaction_new_fwd - gaction_new_bwd) / 2deltaH
        tc_symm_diff = (topcharge_new_fwd - topcharge_new_bwd) / 2deltaH
        ga_symm_diff_smeared = (gaction_new_fwd_smeared - gaction_new_bwd_smeared) / 2deltaH
        tc_symm_diff_smeared = (topcharge_new_fwd_smeared - topcharge_new_bwd_smeared) / 2deltaH

        relerrors[1, group_direction] = (ga_symm_diff - dgaction_proj) / ga_symm_diff
        relerrors[2, group_direction] =
            (ga_symm_diff_smeared - dgaction_proj_smeared) / ga_symm_diff_smeared
        relerrors[3, group_direction] = (tc_symm_diff - dtopcharge_proj) / tc_symm_diff
        relerrors[4, group_direction] =
            (tc_symm_diff_smeared - dtopcharge_proj_smeared) / tc_symm_diff_smeared

        println("================= Group direction $(group_direction) =================")

        # println("gaction before: ", gaction_old)
        # println("gaction fwd: ", gaction_new_fwd)
        # println("gaction bwd: ", gaction_new_bwd)
        # println("/ GA Difference: \t", (gaction_new_fwd - gaction_old) / deltaH)
        # println("/ GA Symm. Diff: \t", ga_symm_diff)
        # println("/ GA Derivative: \t", dgaction_proj)
        println(
            "/ GA rel. error (unsmeared): \t",
            (ga_symm_diff - dgaction_proj) / ga_symm_diff
        )
        println(
            "/ GA rel. error (smeared):   \t",
            (ga_symm_diff_smeared - dgaction_proj_smeared) / ga_symm_diff_smeared
        )

        println("")

        # println("topcharge before: ", topcharge_old)
        # println("topcharge fwd: ", topcharge_new_fwd)
        # println("topcharge bwd: ", topcharge_new_bwd)
        # println("/ TC Difference: \t", (topcharge_new_fwd - topcharge_old) / deltaH)
        # println("/ TC Symm. Diff: \t", tc_symm_diff)
        # println("/ TC Derivative: \t", dtopcharge_proj)
        println(
            "/ TC rel. error (unsmeared): \t",
            (tc_symm_diff - dtopcharge_proj) / tc_symm_diff
        )
        println(
            "/ TC rel. error (smeared):   \t",
            (tc_symm_diff_smeared - dtopcharge_proj_smeared) / tc_symm_diff_smeared
        )
    end

    return relerrors
end

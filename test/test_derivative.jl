function SU3testderivative()
    Random.seed!(1206)
    println("SU3testderivative")
    NX = 4; NY = 4; NZ = 4; NT = 4;
    U = MetaQCD.initial_gauges("hot", NX, NY, NZ, NT, 6.0, WilsonGaugeAction)

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
        topcharge_new_fwd = top_charge(Clover(), Ufwd)

        Ubwd = deepcopy(U)
        Ubwd[direction][site] = expλ(group_direction, -deltaH) * Ubwd[direction][site]
        gaction_new_bwd = calc_gauge_action(Ubwd)
        topcharge_new_bwd = top_charge(Clover(), Ubwd)

        # Smeared
        Ufwd = deepcopy(U)
        Ufwd[direction][site] = expλ(group_direction, deltaH) * Ufwd[direction][site]
        calc_smearedU!(smearing, Ufwd)
        gaction_new_fwd_smeared = calc_gauge_action(smearing.Usmeared_multi[end])
        topcharge_new_fwd_smeared = top_charge(Clover(), smearing.Usmeared_multi[end])

        Ubwd = deepcopy(U)
        Ubwd[direction][site] = expλ(group_direction, -deltaH) * Ubwd[direction][site]
        calc_smearedU!(smearing, Ubwd)
        gaction_new_bwd_smeared = calc_gauge_action(smearing.Usmeared_multi[end])
        topcharge_new_bwd_smeared = top_charge(Clover(), smearing.Usmeared_multi[end])

        calc_dSdU_bare!(dSdU, staples, U, nothing, NoSmearing())
        calc_dSdU_bare!(dSdU_smeared, staples, U, temp_force, smearing)
        calc_dQdU_bare!(dQdU, fieldstrength, U, Clover())
        calc_dQdU_bare!(dQdU_smeared, fieldstrength, U, temp_force, Clover(), smearing)

        dgaction_proj = real(multr(im * λ[group_direction], dSdU[direction][site]))
        dtopcharge_proj = real(multr(im * λ[group_direction], dQdU[direction][site]))
        dgaction_proj_smeared = real(multr(im * λ[group_direction], dSdU_smeared[direction][site]))
        dtopcharge_proj_smeared = real(multr(im * λ[group_direction], dQdU_smeared[direction][site]))

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
        println("/ GA rel. error (unsmeared): \t",
                (ga_symm_diff - dgaction_proj) / ga_symm_diff)
        println("/ GA rel. error (smeared):   \t",
                (ga_symm_diff_smeared - dgaction_proj_smeared) / ga_symm_diff_smeared)
        println("")
        println("/ TC rel. error (unsmeared): \t",
                (tc_symm_diff - dtopcharge_proj) / tc_symm_diff)
        println("/ TC rel. error (smeared):   \t",
                (tc_symm_diff_smeared - dtopcharge_proj_smeared) / tc_symm_diff_smeared)
    end

    return relerrors
end

function calc_dQdU_bare!(dQdU, F, U, temp_force, kind_of_charge, smearing)
    calc_smearedU!(smearing, U)
    fully_smeared_U = smearing.Usmeared_multi[end]
    calc_dQdU_bare!(dQdU, F, fully_smeared_U, kind_of_charge)
    stout_backprop!(dQdU, temp_force, smearing)
    return nothing
end

function calc_dQdU_bare!(dQdU, F, U, kind_of_charge)
    fieldstrength_eachsite!(kind_of_charge, F, U)

    @batch for site in eachindex(U)
        tmp1 = cmatmul_oo(U[1][site], (
            ∇trFμνFρσ(kind_of_charge, U, F, 1, 2, 3, 4, site) -
            ∇trFμνFρσ(kind_of_charge, U, F, 1, 3, 2, 4, site) +
            ∇trFμνFρσ(kind_of_charge, U, F, 1, 4, 2, 3, site)
        ))
        dQdU[1][site] = 1/4π^2 * traceless_antihermitian(tmp1)

        tmp2 = cmatmul_oo(U[2][site], (
            ∇trFμνFρσ(kind_of_charge, U, F, 2, 3, 1, 4, site) -
            ∇trFμνFρσ(kind_of_charge, U, F, 2, 1, 3, 4, site) -
            ∇trFμνFρσ(kind_of_charge, U, F, 2, 4, 1, 3, site)
        ))
        dQdU[2][site] = 1/4π^2 * traceless_antihermitian(tmp2)

        tmp3 = cmatmul_oo(U[3][site], (
            ∇trFμνFρσ(kind_of_charge, U, F, 3, 1, 2, 4, site) -
            ∇trFμνFρσ(kind_of_charge, U, F, 3, 2, 1, 4, site) +
            ∇trFμνFρσ(kind_of_charge, U, F, 3, 4, 1, 2, site)
        ))
        dQdU[3][site] = 1/4π^2 * traceless_antihermitian(tmp3)

        tmp4 = cmatmul_oo(U[4][site], (
            ∇trFμνFρσ(kind_of_charge, U, F, 4, 2, 1, 3, site) -
            ∇trFμνFρσ(kind_of_charge, U, F, 4, 1, 2, 3, site) -
            ∇trFμνFρσ(kind_of_charge, U, F, 4, 3, 1, 2, site)
        ))
        dQdU[4][site] = 1/4π^2 * traceless_antihermitian(tmp4)
    end

    return nothing
end

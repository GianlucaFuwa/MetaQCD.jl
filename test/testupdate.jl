function SU3testupdate()
    Random.seed!(1206)

    println("SU3testupdate")
    NX = 12; NY = 12; NZ = 12; NT = 12
    U = random_gauges(NX, NY, NZ, NT, 6.0, WilsonGaugeAction);
    #filename = "./test/testconf.txt"
    #load_BridgeText!(filename,U)

    verbose = Verbose2()

    update_method = "hmc"
    kind_of_bias = "none"
    metro_ϵ = 0.1
    metro_multi_hit = 1
    metro_target_acc = 0.5
    hmc_integrator = "OMF4"
    hmc_Δτ = 0.1
    hmc_friction = π/4
    hmc_steps = 10
    hmc_numsmear = 5
    hmc_ρstout = 0.125
    hb_eo = false
    hb_MAXIT = 10
    hb_numHB = 1
    hb_numOR = 4

    updatemethod = Updatemethod(
        U,
        update_method,
        1,
        "",
        kind_of_bias,
        metro_ϵ,
        metro_multi_hit,
        metro_target_acc,
        hmc_integrator,
        hmc_steps,
        hmc_Δτ,
        hmc_friction,
        hmc_numsmear,
        hmc_ρstout,
        hb_MAXIT,
        hb_numHB,
        hb_eo,
        hb_numOR,
    );

    for _ = 1:10
        _, runtime = @timed update!(updatemethod, U, verbose, metro_test=false)
        println("Elapsed time: $runtime [s]")
    end

    numaccepts = 0
    nsweeps = 10

    for _ = 1:nsweeps
        value, runtime = @timed update!(updatemethod, U, verbose, metro_test=true)
        println("Elapsed time: $runtime [s]")
        numaccepts += value
    end

    if update_method != "hmc"
        Sg_final_unsmeared = U.Sg
        println("Final Gauge Action is: ", Sg_final_unsmeared)
    else
        if typeof(updatemethod.smearing) == NoSmearing
            Sg_final_unsmeared = U.Sg
            println("Final Gauge Action is: ", Sg_final_unsmeared)
        else
            Sg_final_unsmeared = U.Sg
            println("Final Gauge Action is: ", Sg_final_unsmeared)
            calc_smearedU!(updatemethod.smearing, U)
            fully_smeared_U = updatemethod.smearing.Usmeared_multi[end]
            Sg_final_smeared = calc_gauge_action(fully_smeared_U)
            println("Final smeared Gauge Action is: ", Sg_final_smeared)
        end
    end

    println("Acceptance Rate: ", 100 * numaccepts / nsweeps, " %")
    return numaccepts / nsweeps
end

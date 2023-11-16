function SU3testupdate()
    Random.seed!(1206)

    println("SU3testupdate")
    NX = 4; NY = 4; NZ = 4; NT = 4
    U = random_gauges(NX, NY, NZ, NT, 6.0, SymanzikTreeGaugeAction);
    #filename = "./test/testconf.txt"
    #load_BridgeText!(filename,U)

    verbose = Verbose2()

    update_method = "hmc"
    kind_of_bias = "none"
    metro_ϵ = 0.1
    metro_numhits = 1
    metro_target_acc = 0.5
    hmc_integrator = "OMF4"
    hmc_trajectory = 1
    hmc_friction = π/2
    hmc_steps = 5
    hmc_numsmear = 5
    hmc_ρstout = 0.12
    hb_eo = false
    hb_MAXIT = 10
    numheatbath = 1
    or_algorithm = "kenney-laub"
    numorelax = 4

    updatemethod = Updatemethod(
        U,
        update_method,
        verbose,
        1,
        "",
        kind_of_bias,
        metro_ϵ,
        metro_numhits,
        metro_target_acc,
        hmc_integrator,
        hmc_steps,
        hmc_trajectory,
        hmc_friction,
        hmc_numsmear,
        hmc_ρstout,
        hb_MAXIT,
        numheatbath,
        hb_eo,
        or_algorithm,
        numorelax,
    );

    println(typeof(updatemethod)) # To check if we are using the right iterator

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

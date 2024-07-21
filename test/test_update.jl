function test_update(
    backend=CPU;
    update_method="heatbath",
    or_algorithm="subgroups",
    hmc_integrator="OMF4",
    hmc_numsmear=0,
    gaction=WilsonGaugeAction,
)
    Random.seed!(123)

    println("SU3testupdate")
    NX = NY = NZ = NT = 4
    U = Gaugefield{backend,Float64,gaction}(NX, NY, NZ, NT, 6.0)
    random_gauges!(U)
    if backend !== nothing
        U = MetaQCD.to_backend(backend, U)
    end

    verbose_level = 2
    kind_of_bias = "none"
    metro_ϵ = 0.2
    metro_numhits = 1
    metro_target_acc = 0.5
    hmc_trajectory = 1
    hmc_friction = 0
    hmc_steps = 5
    hmc_ρstout = 0.12
    hb_eo = true
    hb_MAXIT = 10
    numheatbath = 1
    numorelax = 4

    updatemethod = Updatemethod(
        U,
        update_method,
        verbose_level,
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

    println(typeof(updatemethod), "\n") # To check if we are using the right iterator
    println("Starting action is: $(calc_gauge_action(U))")

    for _ = 1:10
        _, runtime = @timed update!(updatemethod, U, metro_test=false)
        println("Elapsed time: $runtime [s]")
    end

    numaccepts = 0
    nsweeps = 10

    for _ = 1:nsweeps
        value, runtime = @timed update!(updatemethod, U, metro_test=true)
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

    println("Acceptance Rate: ", 100 * numaccepts / nsweeps, " %\n")
    return true
end

function test_update(
    backend=CPU;
    update_method="heatbath",
    or_algorithm="subgroups",
    hmc_integrator="OMF4",
    hmc_numsmear_gauge=0,
    gaction=WilsonGaugeAction,
    nprocs_cart=(1, 1, 1, 1),
    halo_width=1,
)
    Random.seed!(123)

    mpi_amroot() && println("Update algorithm tests")
    MetaQCD.MetaIO.set_global_logger!(1, nothing; tc=true)
    NX = NY = NZ = NT = 4
    U = Gaugefield{CPU,Float64,gaction}(NX, NY, NZ, NT, 6.0, nprocs_cart, halo_width)
    random_gauges!(U)

    if backend !== CPU
        U = MetaQCD.to_backend(backend, U)
    end

    kind_of_bias = "none"
    metro_ϵ = 0.2
    metro_numhits = 1
    metro_target_acc = 0.5
    hmc_trajectory = 1
    hmc_friction = 0
    hmc_steps = 5
    hmc_rhostout_gauge = 0.12
    hb_maxit = 10
    numheatbath = 1
    numorelax = 4

    updatemethod = Updatemethod(
        U, update_method;
        kind_of_bias=kind_of_bias,
        metro_ϵ=metro_ϵ,
        metro_numhits=metro_numhits,
        metro_target_acc=metro_target_acc,
        hmc_integrator=hmc_integrator,
        hmc_steps=hmc_steps,
        hmc_trajectory=hmc_trajectory,
        hmc_friction=hmc_friction,
        hmc_numsmear_gauge=hmc_numsmear_gauge,
        hmc_rhostout_gauge=hmc_rhostout_gauge,
        hb_maxit=hb_maxit,
        numheatbath=numheatbath,
        or_algorithm=or_algorithm,
        numorelax=numorelax,
    );

    mpi_amroot() && println(typeof(updatemethod), "\n") # To check if we are using the right iterator
    mpi_amroot() && println("Starting action is: $(calc_gauge_action(U))")

    for _ in 1:10
        _, runtime = @timed update!(updatemethod, U, metro_test=false)
        println("Elapsed time: $runtime [s]")
    end

    numaccepts = 0
    nsweeps = 10

    for _ in 1:nsweeps
        value, runtime = @timed update!(updatemethod, U, metro_test=true)
        println("Elapsed time: $runtime [s]")
        numaccepts += value
    end

    if update_method != "hmc"
        Sg_final_unsmeared = U.Sg
        println("Final Gauge Action is: ", Sg_final_unsmeared)
    else
        if typeof(updatemethod.smearing_gauge) == NoSmearing
            Sg_final_unsmeared = U.Sg
            mpi_amroot() && println("Final Gauge Action is: ", Sg_final_unsmeared)
        else
            Sg_final_unsmeared = U.Sg
            mpi_amroot() && println("Final Gauge Action is: ", Sg_final_unsmeared)
            calc_smearedU!(updatemethod.smearing_gauge, U)
            fully_smeared_U = updatemethod.smearing_gauge.Usmeared_multi[end]
            Sg_final_smeared = calc_gauge_action(fully_smeared_U)
            mpi_amroot() && println("Final smeared Gauge Action is: ", Sg_final_smeared)
        end
    end

    mpi_amroot() && println("Acceptance Rate: ", 100 * numaccepts / nsweeps, " %\n")
    mpi_barrier()
    return true
end

using MetaQCD
using MetaQCD.Utils
using Random

function test_update(
    backend=CPU;
    update_method="hmc",
    or_algorithm="subgroups",
    hmc_integrator="OMF4",
    hmc_numsmear_gauge=0,
    gaction=WilsonGaugeAction,
    numprocs_cart=(1, 1, 1, 1),
    halo_width=1,
)
    Random.seed!(123)

    mpi_amroot() && println("Update algorithm tests")
    MetaQCD.MetaIO.set_global_logger!(1, nothing; tc=true)
    str = if update_method == "hmc"
        "$(update_method) ($(hmc_integrator))"
    else
        "$(update_method)"
    end
    @testset "$(str)" begin
        NX = NY = NZ = NT = 12
        U = Gaugefield{backend,Float64,gaction,12}(
            NX, NY, NZ, NT, 6.0; numprocs_cart, halo_width
        )
        random_gauges!(U)

        metro_ϵ = 0.2
        metro_numhits = 1
        metro_target_acc = 0.5
        hmc_trajectory = 1
        hmc_friction = 0
        hmc_rhostout_gauge = 0.12
        hb_maxit = 10
        numheatbath = 1
        numorelax = 4

        levels = [Dict(
            "integrator" => hmc_integrator,
            "forces" => [1],
            "numsteps" => 10,
        )]

        updatemethod = Updatemethod(
            U,
            update_method;
            hmc_levels=levels,
            metro_ϵ=metro_ϵ,
            metro_numhits=metro_numhits,
            metro_target_acc=metro_target_acc,
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
        Sg0 = calc_gauge_action(U)
        mpi_amroot() && println("Starting action is: $(Sg0)")

        for _ in 1:10
            _, runtime = @timed update!(updatemethod, U; metro_test=false)
            println("Elapsed time: $runtime [s]")
        end

        numaccepts = 0
        nsweeps = 10

        for _ in 1:nsweeps
            value, runtime = @timed update!(updatemethod, U; metro_test=true)
            println("Elapsed time: $runtime [s]")
            numaccepts += value
        end

        if update_method != "hmc"
            println("Final Gauge Action is: ", calc_gauge_action(U))
        else
            if typeof(updatemethod.smearing_gauge) == NoSmearing
                Sgf = calc_gauge_action(U)
                mpi_amroot() && println("Final Gauge Action is: ", Sgf)
            else
                Sgf = calc_gauge_action(U)
                mpi_amroot() && println("Final Gauge Action is: ", Sgf)
                calc_smearedU!(updatemethod.smearing_gauge, U)
                fully_smeared_U = updatemethod.smearing_gauge.Usmeared_multi[end]
                Sg_final_smeared = calc_gauge_action(fully_smeared_U)
                mpi_amroot() && println("Final smeared Gauge Action is: ", Sg_final_smeared)
            end
        end

        mpi_amroot() && println("Acceptance Rate: ", 100 * numaccepts / nsweeps, " %\n")
        mpi_barrier()
        @test true
    end
    return true
end

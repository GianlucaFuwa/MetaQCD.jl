using Random
include("../src/system/MetaQCD.jl")
using .MetaQCD

function SU3testupdate()
    Random.seed!(1206)

    println("SU3testupdate")
    NX = 4; NY = 4; NZ = 4; NT = 4
    action = WilsonGaugeAction
    U = random_gauges(NX, NY, NZ, NT, 5.7, type_of_gaction = action);
    #filename = "./test/testconf.txt"
    #load_BridgeText!(filename,U)

    verbose = Verbose2()

    update_method = "hmc"
    meta_enabled = false
    metro_ϵ = 0.1
    metro_multi_hit = 1
    metro_target_acc = 0.5
    hmc_integrator = "OMF4"
    hmc_Δτ = 0.1
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
        meta_enabled,
        metro_ϵ,
        metro_multi_hit,
        metro_target_acc,
        hmc_integrator,
        hmc_steps,
        hmc_Δτ,
        hmc_numsmear,
        hmc_ρstout,
        hb_eo,
        hb_MAXIT,
        hb_numHB,
        hb_numOR,
    )
    
    for itrj = 1:10
        value, runtime = @timed update!(updatemethod, U, verbose, metro_test=false)
        normalize!(U)
        println("Elapsed time: $runtime [s]")
    end

    numaccepts = 0
    nsweeps = 1

    for itrj = 1:nsweeps
        value, runtime = @timed update!(updatemethod, U, verbose, metro_test=true)
        normalize!(U)
        println("Elapsed time: $runtime [s]")
        numaccepts += value
    end
    
    if update_method != "hmc"
        Sg_final = calc_gauge_action(U)
    else
        if typeof(updatemethod.smearing) == NoSmearing
            Sg_final = calc_gauge_action(U)
        else
            calc_smearedU!(updatemethod.smearing, U)
            fully_smeared_U = updatemethod.smearing.Usmeared_multi[end]
            Sg_final = calc_gauge_action(fully_smeared_U)
        end
    end

    println("Final Gauge Action is: ", Sg_final)
    println("Acceptance Rate: ", 100 * numaccepts / nsweeps, " %")
    return nothing
end
@time SU3testupdate()

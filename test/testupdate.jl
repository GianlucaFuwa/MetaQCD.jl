using Random
#include("../src/system/MetaQCD.jl")
#using .MetaQCD
function SU3testupdate()
    println(SU3testupdate)
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    action = "wilson"
    U = identity_gauges(NX, NY, NZ, NT, 5.7, kind_of_gaction = action);
    #filename = "./test/testconf.txt"
    #load_BridgeText!(filename,U)

    rng = Xoshiro(1206)
    verbose = Verbose2()

    update_method = "hmc"
    Δτ = 0.1
    ϵ_metro = 0.1
    hmc_steps = 10
    integrator = "OMF4"
    updatemethod = Updatemethod(
        U,
        update_method,
        ϵ_metro,
        1,
        0.5,
        Δτ,
        hmc_steps,
        integrator,
        10,
        4,
        0,
        false,
    );
    
    for itrj = 1:10
        value, runtime = @timed update!(updatemethod, U, rng, verbose, metro_test=false)
        normalize!(U)
        println("Elapsed time: $runtime [s]")
    end

    numaccepts = 0
    nsweeps = 10

    for itrj = 1:nsweeps
        value, runtime = @timed update!(updatemethod, U, rng, verbose, metro_test=true)
        normalize!(U)
        println("Elapsed time: $runtime [s]")
        numaccepts += value
    end

    println("Thermalized Gauge Action is: ", calc_gauge_action(U))
    println("Acceptance Rate: ", 100 * numaccepts / nsweeps, " %")
    return nothing
end
#@time SU3testupdate();
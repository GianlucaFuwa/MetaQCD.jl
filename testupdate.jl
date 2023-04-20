using Random
#include("../src/system/MetaQCD.jl")
#using .MetaQCD
function SU3testupdate()
    println(SU3testupdate)
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    action = "Wilson"
    U = RandomGauges(NX, NY, NZ, NT, 5.7, gaction = action)
    println(U[1][1,1,1,1])
    recalc_GaugeAction!(U)
    #filename = "./test/testconf.txt"
    #load_BridgeText!(filename,U)
    P = Liefield(U)

    rng = Xoshiro(1206)
    verbose = Verbose_2()

    update_method = "Heatbath"
    ϵ = 0.1
    Δτ = 0.1
    hmc_steps = 10
    integrator = "OMF4"
    updatemethod = Updatemethod(U, P, update_method, ϵ, Δτ, hmc_steps, integrator, false)
    for itrj = 1:10
        value, runtime = @timed update!(updatemethod, U, rng, verbose, metro_test=false)
        normalize!(U)
        println("Elapsed time: $runtime [s]")
    end
    numaccepts = 0
    nsweeps = 50
    for itrj = 1:nsweeps
        value, runtime = @timed update!(updatemethod, U, rng, verbose, metro_test=true)
        normalize!(U)
        println("Elapsed time: $runtime [s]")
        numaccepts += value
    end
    recalc_GaugeAction!(U)
    println("Thermalized Gauge Action is: ", U.Sg)
    println("Acceptance Rate: ", 100*numaccepts/nsweeps, " %")
    return U, P
end
U, P = SU3testupdate()
println("Done")
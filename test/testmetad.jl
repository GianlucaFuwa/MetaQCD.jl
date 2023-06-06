using Random
include("../src/system/MetaQCD.jl")
using .MetaQCD
function SU3testmeta()
    Random.seed!(1206)

    println("SU3testmeta")
    NX = 4; NY = 4; NZ = 4; NT = 4
    action = WilsonGaugeAction
    U = random_gauges(NX, NY, NZ, NT, 5.7, type_of_gaction = action);
    #filename = "./test/testconf.txt"
    #load_BridgeText!(filename,U)

    verbose = Verbose2()

    update_method = "hmc"
    meta_enabled = true
    metro_ϵ = 0.1
    metro_multi_hit = 1
    metro_target_acc = 0.5
    hmc_integrator = "OMF4"
    hmc_Δτ = 0.1
    hmc_steps = 10
    hmc_numsmear = 0
    hmc_ρstout = 0.125
    hb_eo = false
    hb_MAXIT = 10
    hb_numHB = 1
    hb_numOR = 4

    bias_kind_of_cv = "clover"
    bias_numsmear = 5
    bias_ρstout = 0.12
    bias_symmetric = true
    bias_CVlims = (-5, 5)
    bias_bin_width = 1e-2
    bias_weight = 1e-2
    bias_penalty_weight = 1000

    Bias = BiasPotential(
        U,
        bias_kind_of_cv,
        bias_numsmear,
        bias_ρstout,
        bias_symmetric,
        bias_CVlims,
        bias_bin_width,
        bias_weight,
        bias_penalty_weight,
    );

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
    
    for _ = 1:1
        value, runtime = @timed update!(
            updatemethod,
            U,
            verbose,
            Bias = Bias,
            metro_test = false,
        )
        normalize!(U)
        println("Elapsed time: $runtime [s]")
    end

    numaccepts = 0
    nsweeps = 0

    for _ = 1:nsweeps
        value, runtime = @timed update!(
            updatemethod,
            U,
            verbose,
            Bias = Bias,
            metro_test = true,
        )
        normalize!(U)
        println("Elapsed time: $runtime [s]")
        numaccepts += value
    end
    
    calc_smearedU!(Bias.smearing, U)
    fully_smeared_U = Bias.smearing.Usmeared_multi[end]
    Q_final = top_charge(fully_smeared_U, bias_kind_of_cv)

    println("Thermalized charge is: ", Q_final)
    println("Acceptance Rate: ", 100 * numaccepts / nsweeps, " %")
    return nothing
end
@time SU3testmeta()
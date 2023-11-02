using MetaQCD
using Random

function SU3testreversibility()
    println("SU3testreversibility")
    Random.seed!(1206)
    N = 4
    Uw = identity_gauges(N, N, N, N, 6.0, DBW2GaugeAction)
    Ud = identity_gauges(N, N, N, N, 6.0, DBW2GaugeAction)
    # if it works for dbw2 it should(!) work for the other improved actions
    loadU_bridge!(Uw, "./test/testconf.txt")
    loadU_bridge!(Ud, "./test/testconf.txt")

    hmc_integrator = ["Leapfrog", "OMF2Slow", "OMF2", "OMF4Slow", "OMF4"]
    hmc_trajectory = 1
    hmc_steps = 10

    biasW = Bias(
        Clover(),
        StoutSmearing(Uw, 5, 0.12),
        true,
        Parametric((-5, 5), 10, 0, 100, 1.4),
        nothing,
        0,
        nothing,
        nothing,
    )
    biasD = Bias(
        Clover(),
        StoutSmearing(Ud, 5, 0.12),
        true,
        Parametric((-5, 5), 10, 0, 100, 1.4),
        nothing,
        0,
        nothing,
        nothing,
    )

    for itg in hmc_integrator
        hmcW =  HMC(Uw, itg, hmc_steps, hmc_trajectory)
        hmcWB = HMC(Uw, itg, hmc_steps, hmc_trajectory, bias_enabled=true)
        hmcD =  HMC(Ud, itg, hmc_steps, hmc_trajectory)
        hmcDB = HMC(Ud, itg, hmc_steps, hmc_trajectory, bias_enabled=true)
        reversibility_test(hmcW, Uw, nothing)
        reversibility_test(hmcWB, Uw, biasW)
        reversibility_test(hmcD, Ud, nothing)
        reversibility_test(hmcDB, Ud, biasD)
    end
    return true
end

function reversibility_test(hmc::HMC{TI,TG,TS,TB}, U, bias, prec=1e-2) where {TI,TG,TS,TB}
    gaussian_momenta!(hmc.P, hmc.friction)
    Sg_old = calc_gauge_action(U)
    trP²_old = -calc_kinetic_energy(hmc.P)
    CV_old = bias≡nothing ? 0.0 : calc_CV(U, bias)
    V_old = bias≡nothing ? 0.0 : bias(CV_old)
    H_old = Sg_old + trP²_old + V_old

    evolve!(TI(), U, hmc, bias)
    normalize!(U)
    mul!(hmc.P, -1)
    evolve!(TI(), U, hmc, bias)
    normalize!(U)

    Sg_new = calc_gauge_action(U)
    trP²_new = -calc_kinetic_energy(hmc.P)
    CV_new = bias≡nothing ? 0.0 : calc_CV(U, bias)
    V_new = bias≡nothing ? 0.0 : bias(CV_new)
    H_new = Sg_new + trP²_new + V_new

    ΔH = H_new - H_old
    @assert !isnan(ΔH) "Reversibility test failed for $(typeof(hmc)) and $(typeof(U)) with ΔH = $(ΔH)"
    return true
end

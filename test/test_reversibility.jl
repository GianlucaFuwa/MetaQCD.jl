using MetaQCD
using Random

function SU3testreversibility()
    println("SU3testreversibility")
    Random.seed!(1206)
    N = 4
    U = identity_gauges(N, N, N, N, 6.0, WilsonGaugeAction)
    # if it works for dbw2 it should(!) work for the other improved actions
    loadU_bridge!(U, "./test/testconf.txt")

    hmc_integrator = ["Leapfrog", "OMF2Slow", "OMF2", "OMF4Slow", "OMF4"]
    hmc_trajectory = 1
    hmc_steps = 10

    bias = Bias(
        Clover(),
        StoutSmearing(Uw, 5, 0.12),
        true,
        Parametric((-5, 5), 10, 0, 100, 1.4),
        nothing,
        0,
        nothing,
        nothing,
    )

    for itg in hmc_integrator
        hmc =  HMC(U, itg, hmc_steps, hmc_trajectory)
        hmcB = HMC(U, itg, hmc_steps, hmc_trajectory, bias_enabled=true)

        reversibility_test(hmc, U, nothing)
        reversibility_test(hmcB, U, bias)
    end
    return true
end

function reversibility_test(hmc::HMC{TI,TG,TS}, U, bias, prec=1e-2) where {TI,TG,TS}
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
    println("ΔH = ", ΔH)
    @assert !isnan(ΔH) "Reversibility test failed for $(typeof(hmc)) and $(typeof(U)) with ΔH = $(ΔH)"
    return true
end

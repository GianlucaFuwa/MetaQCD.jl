using MetaQCD
using Random
using Printf

function SU3testreversibility(;
    integrator=Leapfrog, gaction=WilsonGaugeAction, faction=nothing, with_bias=false
)
    println("┌ Testing reversibility of $integrator")
    println("|  gauge action: $gaction")
    println("|  fermion action: $faction")
    println("|  bias enabled: $(with_bias)")
    println("└\n")
    MetaQCD.Output.set_global_logger!(1, devnull; tc=true)
    Random.seed!(123)
    N = 4
    U = Gaugefield(N, N, N, N, 6.0; GA=gaction)
    fermion = if faction === nothing
        nothing
    else
        (faction(U, 0.1),)
    end
    # if it works for dbw2 it should(!) work for the other improved actions
    loadU!(BridgeFormat(), U, "./test/testconf.txt")

    hmc_trajectory = 1
    hmc_steps = 10

    bias = if with_bias
        Bias(
            Clover(),
            StoutSmearing(U, 4, 0.12),
            true,
            Parametric((-5, 5), 10, 0, 100, 1.4),
            nothing,
            0,
            nothing,
            nothing,
        )
    else
        nothing
    end

    hmc = HMC(U, integrator, hmc_trajectory, hmc_steps; fermion_action=faction)
    hmcB = if with_bias
        HMC(U, integrator, hmc_trajectory, hmc_steps; bias_enabled=true)
    else
        nothing
    end

    dH_n = reversibility_test(hmc, U, fermion, nothing, "without bias")
    dH_b = with_bias ? reversibility_test(hmcB, U, fermion, bias, "with bias") : nothing

    return dH_n, dH_b
end

function reversibility_test(hmc::HMC{TI}, U, fermion, bias, str) where {TI}
    isnothing(fermion) ? nothing : sample_pseudofermions!(hmc.ϕ[1], fermion[1], U)
    gaussian_TA!(hmc.P, hmc.friction)
    Sg_old = calc_gauge_action(U)
    trP²_old = -calc_kinetic_energy(hmc.P)
    Sf_old = isnothing(fermion) ? 0.0 : calc_fermion_action(fermion[1], U, hmc.ϕ[1])
    CV_old = isnothing(bias) ? 0.0 : calc_CV(U, bias)
    V_old = isnothing(bias) ? 0.0 : bias(CV_old)
    H_old = Sg_old + trP²_old + Sf_old + V_old

    evolve!(TI(), U, hmc, fermion, bias)
    MetaQCD.normalize!(U)
    MetaQCD.Gaugefields.mul!(hmc.P, -1)
    evolve!(TI(), U, hmc, fermion, bias)
    MetaQCD.normalize!(U)

    Sg_new = calc_gauge_action(U)
    trP²_new = -calc_kinetic_energy(hmc.P)
    Sf_new = isnothing(fermion) ? 0.0 : calc_fermion_action(fermion[1], U, hmc.ϕ[1])
    CV_new = isnothing(bias) ? 0.0 : calc_CV(U, bias)
    V_new = isnothing(bias) ? 0.0 : bias(CV_new)
    H_new = Sg_new + trP²_new + Sf_new + V_new

    ΔH = H_new - H_old
    println("┌ $str")
    @printf("| ΔSg   = %+.5e (%+.10e)\n", Sg_new - Sg_old, Sg_old)
    @printf("| ΔtrP² = %+.5e (%+.10e)\n", trP²_new - trP²_old, trP²_old)
    @printf("| ΔSf   = %+.5e (%+.10e)\n", Sf_new - Sf_old, Sf_old)
    @printf("| ΔV    = %+.5e (%+.10e)\n", V_new - V_old, V_old)
    @printf("| ΔH    = %+.5e (%+.10e)\n", ΔH, H_old)
    @printf("└\n")
    return ΔH
end

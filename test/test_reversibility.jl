using MetaQCD
using Random
using Printf
using TimerOutputs

function test_reversibility(
    ;
    integrator=Leapfrog(), 
    gaction=WilsonGaugeAction,
    faction=QuenchedFermionAction,
    Nf=2,
    with_bias=false,
    hmc_trajectory=1,
    hmc_steps=100,
)
    MetaQCD.MetaIO.set_global_logger!(1, nothing; tc=true)
    Random.seed!(123)
    N = 12
    U = Gaugefield{CPU,Float64,gaction}(N, N, N, N, 6.0)
    fermion = if isnothing(faction) || faction === QuenchedFermionAction
        QuenchedFermionAction()
    else
        (faction(U, 0.01; Nf=Nf, cg_tol_md=1e-7, cg_tol_action=1e-9),)
    end
    println("┌ Testing reversibility of $integrator")
    println("|  gauge action: $gaction")
    println("|  fermion action: $faction")
    println("|  bias enabled: $(with_bias)")
    println("└\n")

    to = TimerOutput()

    identity_gauges!(U)
    # @timeit to "load_field!" load_field!(BridgeFormat(), U, "./test/testconf.txt")

    bias = if with_bias
        Bias(
            Clover(),
            StoutSmearing(U; numlayers=4, rho=0.12),
            true,
            Parametric((-5, 5), 10, 0, 100, 1.4),
            nothing,
            nothing,
            nothing,
            0,
        )
    else
        NoBias()
    end

    hmc = HMC(
        U, integrator, hmc_trajectory, hmc_steps; fermion_action=faction
    )
    hmcB = if with_bias
        HMC(U, integrator, hmc_trajectory, hmc_steps; bias_enabled=true)
    else
        nothing
    end

    dH_n = reversibility_test(hmc, U, fermion, NoBias(), "without bias", to)
    dH_b = with_bias ? reversibility_test(hmcB, U, fermion, bias, "with bias", to) : nothing
    # show(to)
    return dH_n, dH_b
end

function reversibility_test(hmc::HMC{TI}, U, fermion, bias, str, to) where {TI}
    @timeit to "sample_pseudofermions!" if isnothing(hmc.ϕ)
        nothing
    else
        sample_pseudofermions!(hmc.ϕ, fermion, U)
    end
    @timeit to "gaussian_TA!" gaussian_TA!(hmc.P, hmc.friction)
    @timeit to "gauge action" Sg_old = calc_gauge_action(U)
    @timeit to "kinetic energy" trP²_old = -calc_kinetic_energy(hmc.P)
    @timeit to "fermion action" Sf_old =
        isnothing(hmc.ϕ) ? 0.0 : calc_fermion_action(fermion, U, hmc.ϕ)
    @timeit to "CV" CV_old = isnothing(bias) ? 0.0 : calc_cv(U, bias)
    V_old = isnothing(bias) ? 0.0 : bias(CV_old)
    H_old = Sg_old + trP²_old + Sf_old + V_old

    @timeit to "evolve!" evolve!(hmc.integrator, U, hmc, fermion, bias)
    @timeit to "normalize!" MetaQCD.normalize!(U)

    CV_mid = isnothing(bias) ? 0.0 : calc_cv(U, bias)
    V_mid = isnothing(bias) ? 0.0 : bias(CV_mid)
    H_mid = calc_gauge_action(U) - calc_kinetic_energy(hmc.P) + V_mid
    ΔH_mid = H_mid - H_old
    @timeit to "invert momenta" MetaQCD.Fields.mul!(hmc.P, -1)

    @timeit to "evolve!" evolve!(hmc.integrator, U, hmc, fermion, bias)
    @timeit to "normalize!" MetaQCD.normalize!(U)

    @timeit to "gauge action" Sg_new = calc_gauge_action(U)
    @timeit to "kinetic energy" trP²_new = -calc_kinetic_energy(hmc.P)
    @timeit to "fermion action" Sf_new =
        isnothing(hmc.ϕ) ? 0.0 : calc_fermion_action(fermion, U, hmc.ϕ)
    @timeit to "CV" CV_new = isnothing(bias) ? 0.0 : calc_cv(U, bias)
    V_new = isnothing(bias) ? 0.0 : bias(CV_new)
    H_new = Sg_new + trP²_new + Sf_new + V_new

    ΔH = H_new - H_old
    println("┌ $str")
    @printf("| ΔSg    = %+.5e (%+.10e)\n", Sg_new - Sg_old, Sg_old)
    @printf("| ΔtrP²  = %+.5e (%+.10e)\n", trP²_new - trP²_old, trP²_old)
    @printf("| ΔSf    = %+.5e (%+.10e)\n", Sf_new - Sf_old, Sf_old)
    @printf("| ΔV     = %+.5e (%+.10e)\n", V_new - V_old, V_old)
    @printf("| ΔH_mid = %+.5e (%+.10e)\n", ΔH_mid, H_old)
    @printf("| ΔH     = %+.5e (%+.10e)\n", ΔH, H_old)
    @printf("└\n")
    return ΔH
end

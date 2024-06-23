using MetaQCD
using Random
using Printf
using TimerOutputs

function SU3testreversibility(;
    integrator=Leapfrog(), gaction=WilsonGaugeAction, faction=nothing, Nf=2, with_bias=false
)
    println("┌ Testing reversibility of $integrator")
    println("|  gauge action: $gaction")
    println("|  fermion action: $faction")
    println("|  bias enabled: $(with_bias)")
    println("└\n")
    MetaQCD.Output.set_global_logger!(1, devnull; tc=true)
    Random.seed!(123)
    N = 4
    U = Gaugefield{CPU,Float64,gaction}(N, N, N, N, 6.0)
    fermion = if faction === nothing
        nothing
    else
        (faction(U, 0.01; Nf=Nf, cg_tol_md=1e-7, cg_tol_action=1e-9),)
    end

    to = TimerOutput()

    @timeit to "loadU!" loadU!(BridgeFormat(), U, "./test/testconf.txt")

    hmc_trajectory = 1
    hmc_steps = 100

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

    hmc = HMC(
        U, integrator, hmc_trajectory, hmc_steps; fermion_action=faction
    )
    hmcB = if with_bias
        HMC(U, integrator, hmc_trajectory, hmc_steps; bias_enabled=true)
    else
        nothing
    end

    dH_n = reversibility_test(hmc, U, fermion, nothing, "without bias", to)
    dH_b = with_bias ? reversibility_test(hmcB, U, fermion, bias, "with bias", to) : nothing
    # show(to)
    return dH_n, dH_b
end

function reversibility_test(hmc::HMC{TI}, U, fermion, bias, str, to) where {TI}
    @timeit to "sample_pseudofermions!" if isnothing(fermion)
        nothing
    else
        sample_pseudofermions!(hmc.ϕ[1], fermion[1], U)
    end
    @timeit to "gaussian_TA!" gaussian_TA!(hmc.P, hmc.friction)
    @timeit to "gauge action" Sg_old = calc_gauge_action(U)
    @timeit to "kinetic energy" trP²_old = -calc_kinetic_energy(hmc.P)
    @timeit to "fermion action" Sf_old =
        isnothing(fermion) ? 0.0 : calc_fermion_action(fermion[1], U, hmc.ϕ[1])
    @timeit to "CV" CV_old = isnothing(bias) ? 0.0 : calc_CV(U, bias)
    V_old = isnothing(bias) ? 0.0 : bias(CV_old)
    H_old = Sg_old + trP²_old + Sf_old + V_old

    @timeit to "evolve!" evolve!(hmc.integrator, U, hmc, fermion, bias)
    @timeit to "normalize!" MetaQCD.normalize!(U)

    @show (calc_gauge_action(U) - calc_kinetic_energy(hmc.P)) - H_old
    @timeit to "invert momenta" MetaQCD.Fields.mul!(hmc.P, -1)

    @timeit to "evolve!" evolve!(hmc.integrator, U, hmc, fermion, bias)
    @timeit to "normalize!" MetaQCD.normalize!(U)

    @timeit to "gauge action" Sg_new = calc_gauge_action(U)
    @timeit to "kinetic energy" trP²_new = -calc_kinetic_energy(hmc.P)
    @timeit to "fermion action" Sf_new =
        isnothing(fermion) ? 0.0 : calc_fermion_action(fermion[1], U, hmc.ϕ[1])
    @timeit to "CV" CV_new = isnothing(bias) ? 0.0 : calc_CV(U, bias)
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

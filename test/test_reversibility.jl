using MetaQCD
using MetaQCD.MetaIO: printf
using Random
using TimerOutputs

function test_reversibility(
    T=Float64,
    backend=CPU;
    cg_float_type=T,
    gaction=SymanzikTreeGaugeAction,
    faction=false,
    Nf=8,
    with_bias=false,
    integrator="OMF4", 
    hmc_trajectory=1,
    hmc_steps=5,
)
    MetaQCD.MetaIO.set_global_logger!(1, nothing)
    Random.seed!(123)
    N = 16
    U = Gaugefield{backend,T,gaction,12}(N, N, N, N, 6.0)
    fermion = if !faction
        QuenchedFermionAction()
    else
        (FermionAction("staggered", U, 0.01; Nf, cg_float_type),)
    end
    U64 = T==Float64 ? U : Gaugefield(U, Float64)
    println("┌ Testing reversibility of $integrator")
    println("|  gauge action: $gaction")
    println("|  fermion action: $faction")
    println("|  bias enabled: $(with_bias)")
    println("└\n")

    to = TimerOutput()

    random_gauges!(U)
    if T!=Float64
        copy!(U64, U)
        MetaQCD.normalize!(U64)
    end
    # @timeit to "load_field!" load_field!(BridgeFormat(), U, "./test/testconf.txt")

    # FIXME:
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

    forces = faction ? [1, 2] : [1]
    lvl = [Dict(
        "forces" => forces, 
        "integrator" => integrator,
        "numsteps" => hmc_steps
    )]
    hmc = HMC(
        U,
        lvl,
        hmc_trajectory;
        fermion_action=faction ? "staggered" : "quenched",
        numfermions=faction ? 1 : 0,
        numcv=with_bias ? 1 : 0,
    )
    hmcB = if with_bias
        forces = faction ? [0, 1, 2] : [0, 1]
        lvl = [Dict("forces" => forces, "integrator" => integrator, "numsteps" => hmc_steps)]
        HMC(U, lvl, hmc_trajectory)
    else
        nothing
    end

    dH_n = reversibility_test(hmc, U, U64, fermion, NoBias(), "without bias", to)
    dH_b = with_bias ? reversibility_test(hmcB, U, U64, fermion, bias, "with bias", to) : nothing
    # show(to)
    return dH_n, dH_b
end

function reversibility_test(hmc::HMC{TI}, U, U64, fermion, bias, str, to) where {TI}
    @timeit to "sample_pseudofermions!" if isnothing(hmc.ϕ)
        nothing
    else
        sample_pseudofermions!(hmc.ϕ, fermion, U, NoSmearing(), false)
    end
    @timeit to "gaussian_TA!" gaussian_TA!(hmc.P, hmc.friction)
    @timeit to "gauge action" Sg_old = calc_gauge_action(U64)
    @timeit to "kinetic energy" trP²_old = -calc_kinetic_energy(hmc.P)
    @timeit to "fermion action" Sf_old = isnothing(hmc.ϕ) ? 0.0 : calc_fermion_action(fermion, U, hmc.ϕ, NoSmearing(), false)
    @timeit to "CV" CV_old = isnothing(bias) ? 0.0 : calc_cv(U, bias)
    V_old = isnothing(bias) ? 0.0 : bias(CV_old)
    H_old = Sg_old + trP²_old + Sf_old + V_old

    @timeit to "evolve!" evolve!(U, hmc, fermion, bias, false, 1)
    if U !== U64
        copy!(U64, U)
        MetaQCD.normalize!(U64)
    end

    Sg_mid = calc_gauge_action(U64)
    trP²_mid = -calc_kinetic_energy(hmc.P)
    Sf_mid = isnothing(hmc.ϕ) ? 0.0 : calc_fermion_action(fermion, U, hmc.ϕ, NoSmearing(), false)
    CV_mid = isnothing(bias) ? 0.0 : calc_cv(U, bias)
    V_mid = isnothing(bias) ? 0.0 : bias(CV_mid)
    H_mid = Sg_mid + trP²_mid + Sf_mid + V_mid
    ΔH_mid = H_mid - H_old

    @timeit to "invert momenta" MetaQCD.Fields.mul!(hmc.P, -1)

    @timeit to "evolve!" evolve!(U, hmc, fermion, bias, false, 1)
    if U !== U64
        copy!(U64, U)
        MetaQCD.normalize!(U64)
    end

    @timeit to "gauge action" Sg_new = calc_gauge_action(U64)
    @timeit to "kinetic energy" trP²_new = -calc_kinetic_energy(hmc.P)
    @timeit to "fermion action" Sf_new = isnothing(hmc.ϕ) ? 0.0 : calc_fermion_action(fermion, U, hmc.ϕ, NoSmearing(), false)
    @timeit to "CV" CV_new = isnothing(bias) ? 0.0 : calc_cv(U, bias)
    V_new = isnothing(bias) ? 0.0 : bias(CV_new)
    H_new = Sg_new + trP²_new + Sf_new + V_new

    ΔH = H_new - H_old
    printf("┌ $(str)\n")
    printf("| ΔSg_mid= $(Sg_mid - Sg_old)\n")
    printf("| ΔSg    = $(Sg_new - Sg_old)\n")
    printf("| ΔtrP²  = $(trP²_new - trP²_old)\n")
    printf("| ΔSf    = $(Sf_new - Sf_old)\n")
    printf("| ΔV     = $(V_new - V_old)\n")
    printf("| ΔH_mid = $(ΔH_mid)\n")
    printf("| ΔH     = $(ΔH)\n")
    printf("└\n")
    # printf("| ΔSg    = %+.5e (%+.10e)\n", Sg_new - Sg_old, Sg_old)
    # printf("| ΔtrP²  = %+.5e (%+.10e)\n", trP²_new - trP²_old, trP²_old)
    # printf("| ΔSf    = %+.5e (%+.10e)\n", Sf_new - Sf_old, Sf_old)
    # printf("| ΔV     = %+.5e (%+.10e)\n", V_new - V_old, V_old)
    # printf("| ΔH_mid = %+.5e (%+.10e)\n", ΔH_mid, H_old)
    # printf("| ΔH     = %+.5e (%+.10e)\n", ΔH, H_old)
    return ΔH
end

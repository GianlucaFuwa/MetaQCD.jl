using MetaQCD
using MetaQCD.BiasModule: Metadynamics, MetadynamicsParameters
using MetaQCD.MetaIO: printf
using MetaQCD.Updates: LeapfrogConstrained, OMF4Constrained
using Random
using TimerOutputs
using Statistics
using Plots
using DelimitedFiles

function test_reversibility(
    T=Float64,
    backend=CPU;
    cg_float_type=T,
    gaction=WilsonGaugeAction,
    faction=false,
    Nf=8,
    with_bias=false,
    integrator="OMF4", 
    hmc_trajectory=1,
    hmc_steps=5,
    beta=6.1912,
    N=14,
    nreps=1,
)
    MetaQCD.MetaIO.set_global_logger!(1, nothing)
    Random.seed!(123)
    U = Gaugefield{backend,T,gaction,12}(N, N, N, N, beta)
    fermion = if !faction
        QuenchedFermionAction()
    else
        (FermionAction("staggered", U, 0.1; Nf, cg_float_type),)
    end
    U64 = T==Float64 ? U : Gaugefield(U, Float64)
    println("┌ Testing reversibility of $integrator")
    println("|  gauge action: $gaction")
    println("|  fermion action: $faction")
    println("|  bias enabled: $(with_bias)")
    println("└\n")

    to = TimerOutput()

    if T!=Float64
        copy!(U64, U)
        MetaQCD.normalize!(U64)
    end
    # @timeit to "load_field!" load_field!(BridgeFormat(), U, "./test/testconf.txt")
    bias_params = MetadynamicsParameters(
        load_bias = ["/home/gialu/ensembles/PP/topsusc_wilson_beta6.1912_14_14_build_7smear/biaspotentials/bias_000.metad"],
        numsmears_for_cv = 7
    )

    # FIXME:
    b = Metadynamics(bias_params)
    bias = if with_bias
        Bias(
            U,
            [7],
            0.12,
            (b,),
            nothing,
            nothing,
            nothing,
            nothing,
            MetaQCD.BiasModule.CVSampler(b)
        )
    else
        NoBias()
    end

    forces = if with_bias
        faction ? [0, 1, 2] : [0, 1]
    else
        faction ? [1, 2] : [1]
    end

    numsteps = [5]
    velocities = [0.1]
    dH = zeros(length(numsteps), length(velocities))
    W = deepcopy(dH)
    # load_field!(BridgeFormat(), U, pkgdir(MetaQCD, "test", "testconf.txt"))
    for is in eachindex(numsteps)
        for iv in eachindex(velocities)
            lvl = [
                Dict(
                    "forces" => forces, 
                    "integrator" => integrator,
                    "numsteps" => numsteps[is],
                ),
            ]
            hmc = HMC(
                U,
                lvl,
                hmc_trajectory;
                velocity=velocities[iv],
                fermion_action=faction ? "staggered" : "quenched",
                numfermions=faction ? 1 : 0,
                numcv=with_bias ? 1 : 0,
            )
            load_field!(BridgeFormat(), U, "/home/gialu/projects/MetaQCD-CHMC/ensembles/constraint_test_14/configs/config_00000005.txt")
            dh = 0.0
            w = 0.0
            for _ in 1:nreps
                dh_i, w_i = reversibility_test(hmc, U, U64, fermion, bias, "", to)
                dh += dh_i
                w += w_i
            end
            dH[is, iv] = dh/nreps
            W[is, iv] = w/nreps
        end
    end
    # @show mean(dH)
    # plot(dH)
    # show(to)
    for is in eachindex(numsteps)
        for iv in eachindex(velocities)
            println("n, v = $(numsteps[is]), $(velocities[iv]):\tdH, W = $(dH[is, iv]), $(W[is, iv])")
        end
    end

    return dH, W
end

function reversibility_test(hmc::HMC{TI}, U, U64, fermion, bias, str, to) where {TI}
    @timeit to "sample_pseudofermions!" if isnothing(hmc.ϕ)
        nothing
    else
        sample_pseudofermions!(hmc.ϕ, fermion, U, NoSmearing(), false)
    end
    @timeit to "gaussian_TA!" gaussian_TA!(hmc.P, hmc.friction)
    MetaQCD.Updates.enforce_hidden_constraint!(hmc, U, bias)
    @timeit to "gauge action" Sg_old = calc_gauge_action(U64)
    @timeit to "kinetic energy" trP²_old = -calc_kinetic_energy(hmc.P)
    @timeit to "fermion action" Sf_old = isnothing(hmc.ϕ) ? 0.0 : calc_fermion_action(fermion, U, hmc.ϕ, NoSmearing(), false)
    @timeit to "CV" CV_old = isnothing(bias) ? 0.0 : calc_cv(U, bias)
    V_old = isnothing(bias) ? 0.0 : bias(CV_old)
    H_old = Sg_old + trP²_old + Sf_old + V_old

    # goal = 0.0+0.83rand()
    if hmc.levels[1].integrator isa LeapfrogConstrained || hmc.levels[1].integrator isa OMF4Constrained
        hmc.levels[1].integrator.interval = (
            CV_old[1],
            MetaQCD.BiasModule.get_sample(bias.sampler)
        )
        Z = hmc.levels[1].integrator.interval
        p = [
            MetaQCD.BiasModule.get_probability(bias.sampler, Z[1]),
            MetaQCD.BiasModule.get_probability(bias.sampler, Z[2])
        ]
        println("start: $(round(Z[1]; sigdigits=4))\tend: $(round(Z[2]; sigdigits=4))")
    else
        p = [1, 1]
    end

    # MetaQCD.Updates.enforce_hidden_constraint!(hmc, U, bias)
    @timeit to "evolve!" W_mid = evolve!(hmc.levels[end].integrator, U, hmc, fermion, bias, true, 1)
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
    ΔSg_mid = Sg_mid - Sg_old
    ΔSf_mid = Sf_mid - Sf_old
    ΔH_mid = H_mid - H_old
    ΔtrP²_mid = trP²_mid - trP²_old
    Δp = log(p[1]) - log(p[2])
    # @show ΔSg_mid
    # @show ΔH_mid
    # @show abs(hmc.levels[1].integrator.interval[2]-CV_mid[1])
    @show ΔH_mid, W_mid, Δp

    # @timeit to "invert momenta" MetaQCD.Fields.mul!(hmc.P, -1)
    # if hmc.levels[1].integrator isa LeapfrogConstrained ||hmc.levels[1].integrator isa OMF4Constrained
    #     itvl = hmc.levels[1].integrator.interval
    #     hmc.levels[1].integrator.interval = (itvl[2], itvl[1])
    #     println(hmc.levels[1].integrator.interval)
    # end
    #
    # @timeit to "evolve!" W = evolve!(hmc.levels[end].integrator, U, hmc, fermion, bias, true, 1)
    # @timeit to "invert momenta" MetaQCD.Fields.mul!(hmc.P, -1)
    # if U !== U64
    #     copy!(U64, U)
    #     MetaQCD.normalize!(U64)
    # end
    #
    # @timeit to "gauge action" Sg_new = calc_gauge_action(U64)
    # @timeit to "kinetic energy" trP²_new = -calc_kinetic_energy(hmc.P)
    # @timeit to "fermion action" Sf_new = isnothing(hmc.ϕ) ? 0.0 : calc_fermion_action(fermion, U, hmc.ϕ, NoSmearing(), false)
    # @timeit to "CV" CV_new = isnothing(bias) ? 0.0 : calc_cv(U, bias)
    # V_new = isnothing(bias) ? 0.0 : bias(CV_new)
    # H_new = Sg_new + trP²_new + Sf_new + V_new 
    #
    # ΔH = H_new - H_old
    # printf("┌ $(str)\n")
    # printf("| ΔSg_mid   = $(ΔSg_mid)\n")
    # printf("| ΔSg       = $(Sg_new - Sg_old)\n")
    # printf("| ΔtrP²     = $(trP²_new - trP²_old)\n")
    # printf("| ΔtrP²_mid = $(ΔtrP²_mid)\n")
    # printf("| ΔSf       = $(Sf_new - Sf_old)\n")
    # printf("| ΔSf_mid   = $(ΔSf_mid)\n")
    # printf("| ΔV        = $(V_new - V_old)\n")
    # printf("| ΔH_mid    = $(ΔH_mid)\n")
    # printf("| ΔH        = $(ΔH)\n")
    # printf("| W         = $(W)\n")
    # printf("| W_mid     = $(W_mid)\n")
    # printf("└\n")
    return ΔH_mid+Δp, W_mid
end

function chmc_scan()
    results_dH = zeros(5, 5)
    results_W = zeros(5, 5)

    for (it, hmc_trajectory) in enumerate([4, 8, 16, 32])
        for (is, hmc_steps) in enumerate([20, 40, 80, 160])
            integrator = "OMF4Constrained"
            dH, W = test_reversibility(
                Float64, CPU; hmc_trajectory, hmc_steps, integrator, with_bias=true
            )
            results_dH[is, it] = dH[1]
            results_W[is, it] = W[1]
        end
    end

    writedlm("chmc_results_dH.txt", results_dH) 
    writedlm("chmc_results_W.txt", results_W) 
end

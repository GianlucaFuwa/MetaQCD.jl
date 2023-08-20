module Updates
    using Base.Threads: @threads, nthreads, threadid
    using LinearAlgebra
    using StaticArrays
    using Polyester
    using Printf
    using ..Output
    using ..Utils

    import ..Gaugefields: AbstractGaugeAction, Gaugefield, Liefield, Temporaryfield
    import ..Gaugefields: add!, calc_gauge_action, calc_kinetic_energy, normalize!,
        fieldstrength_eachsite!, gaussian_momenta!, staple, staple_eachsite!, substitute_U!
    import ..Gaugefields: Plaquette, Clover
    import ..BiasModule: Bias, BiasEnabled, BiasDisabled, calc_CV, ∂V∂Q
    import ..BiasModule: kind_of_cv, update_bias!
    import ..Parameters: ParameterSet
    import ..Smearing: AbstractSmearing, NoSmearing, StoutSmearing
    import ..Smearing: calc_smearedU!, get_layer, stout_backprop!
    import ..Universe: Univ

    abstract type AbstractUpdate end

    include("./hmc.jl")
    include("./metropolis.jl")
    include("./heatbath.jl")
    include("./overrelaxation.jl")
    include("./parity.jl")
    include("./tempering.jl")

    function Updatemethod(
        parameters::ParameterSet,
        U,
    )
        updatemethod = Updatemethod(
            U,
            parameters.update_method,
            parameters.verboselevel,
            parameters.logdir,
            parameters.kind_of_bias,
            parameters.metro_epsilon,
            parameters.metro_multi_hit,
            parameters.metro_target_acc,
            parameters.hmc_integrator,
            parameters.hmc_steps,
            parameters.hmc_deltatau,
            parameters.hmc_numsmear,
            parameters.hmc_rhostout,
            parameters.hb_maxit,
            parameters.numheatbath,
            parameters.eo,
            parameters.numorelax,
        )
            return updatemethod
    end

    function Updatemethod(
        U,
        update_method,
        verboselevel = 1,
        logdir = "",
        kind_of_bias = "none",
        metro_ϵ = 0.1,
        metro_multi_hit = 1,
        metro_target_acc = 0.5,
        hmc_integrator = "Leapfrog",
        hmc_steps = 10,
        hmc_Δτ = 0.1,
        hmc_numsmear = 0,
        hmc_ρstout = 0,
        hb_MAXIT = 1,
        hb_numHB = 1,
        eo = false,
        numOR = 4,
    )
        if update_method == "hmc"
            updatemethod = HMCUpdate(
                U,
                hmc_integrator,
                hmc_steps,
                hmc_Δτ,
                numsmear = hmc_numsmear,
                ρ_stout = hmc_ρstout,
                bias_enabled = kind_of_bias=="none",
                verboselevel = verboselevel,
                logdir = logdir,
            )
        elseif update_method == "metropolis"
            updatemethod = MetroUpdate(
                U,
                eo,
                metro_ϵ,
                metro_multi_hit,
                metro_target_acc,
                numOR,
            )
        elseif update_method == "heatbath"
            updatemethod = HeatbathUpdate(
                U,
                eo,
                hb_MAXIT,
                hb_numHB,
                numOR,
            )
        else
            error("update method $(update_method) is not supported")
        end

        return updatemethod
    end

    update!(::T, U) where {T<:AbstractUpdate} = nothing
    update!(::Nothing, U) = nothing

end

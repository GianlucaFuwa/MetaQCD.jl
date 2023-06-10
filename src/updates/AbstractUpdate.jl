module AbstractUpdateModule
    using Base.Threads: @threads, nthreads, threadid
    using LinearAlgebra
    using StaticArrays
    using Polyester
    using ..Utils
    using ..VerbosePrint

    import ..SystemParameters: Params
    import ..Gaugefields: AbstractGaugeAction, Gaugefield, Liefield, Temporaryfield
    import ..Gaugefields: add!, calc_gauge_action, calc_kinetic_energy,
        fieldstrength_eachsite!, gaussian_momenta!, staple_eachsite!, substitute_U!
    import ..Metadynamics: BiasPotential, MetaEnabled, MetaDisabled, ∂V∂Q, update_bias!
    import ..AbstractSmearingModule: AbstractSmearing, NoSmearing, StoutSmearing
    import ..AbstractSmearingModule: calc_smearedU!, get_layer, stout_backprop!
    import ..AbstractMeasurementModule: top_charge
    import ..UniverseModule: Univ

    abstract type AbstractUpdate end

    include("./hmc.jl")
    include("./metropolis.jl")
    include("./heatbath.jl")

    function Updatemethod(
        parameters::Params,
        U;
        instance = 1,
    )
        updatemethod = Updatemethod(
            U,
            parameters.update_method,
            parameters.meta_enabled,
            parameters.metro_ϵ,
            parameters.metro_multi_hit,
            parameters.metro_target_acc,
            parameters.hmc_integrator,
            parameters.hmc_steps,
            parameters.hmc_Δτ,
            parameters.hmc_numsmear,
            parameters.hmc_ρstout,
            parameters.hb_eo,
            parameters.hb_MAXIT,
            parameters.hb_numHB,
            parameters.hb_numOR,
        )
            return updatemethod
    end

    function Updatemethod(
        U,
        update_method,
        meta_enabled = false,
        metro_ϵ = 0.1,
        metro_multi_hit = 1,
        metro_target_acc = 0.5,
        hmc_integrator = "leapfrog",
        hmc_steps = 10,
        hmc_Δτ = 0.1,
        hmc_numsmear = 0,
        hmc_ρstout = 0,
        hb_eo = false,
        hb_MAXIT = 1,
        hb_numHB = 1,
        hb_numOR = 4,
    )
        updatemethod = HMCUpdate(
            U,
            hmc_integrator,
            hmc_steps,
            hmc_Δτ,
            numsmear = hmc_numsmear,
            ρ_stout = hmc_ρstout,
            meta_enabled = meta_enabled,
        )
        if update_method == "hmc"
            updatemethod = HMCUpdate(
                U,
                hmc_integrator,
                hmc_steps,
                hmc_Δτ,
                numsmear = hmc_numsmear,
                ρ_stout = hmc_ρstout,
                meta_enabled = meta_enabled,
            )
        elseif update_method == "metropolis"
            updatemethod = MetroUpdate(
                U,
                metro_ϵ,
                metro_multi_hit,
                metro_target_acc,
                meta_enabled,
            )
        elseif update_method == "heatbath"
            @assert meta_enabled == false "MetaD can only be used with HMC or Metropolis"
            updatemethod = HeatbathUpdate(
                U,
                hb_eo,
                hb_MAXIT,
                hb_numHB,
                hb_numOR,
            )
        else
            error("update method $(update_method) is not supported")
        end

        return updatemethod
    end

    function update!(updatemethod::T, U) where {T<:AbstractUpdate}
        error("updatemethod type $(typeof(updatemethod)) is not supported")
    end

end

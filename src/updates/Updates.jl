module Updates
    using Base.Threads: @threads, nthreads, threadid
    using LinearAlgebra
    using StaticArrays
    using Polyester
    using ..Output
    using ..Utils

    import ..Gaugefields: AbstractGaugeAction, Gaugefield, Liefield, Temporaryfield
    import ..Gaugefields: add!, calc_gauge_action, calc_kinetic_energy, normalize!,
        fieldstrength_eachsite!, gaussian_momenta!, staple_eachsite!, substitute_U!
    import ..Measurements: top_charge
    import ..Metadynamics: BiasPotential, MetaEnabled, MetaDisabled, ∂V∂Q, update_bias!
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
            parameters.meta_enabled,
            parameters.metro_ϵ,
            parameters.metro_multi_hit,
            parameters.metro_target_acc,
            parameters.hmc_integrator,
            parameters.hmc_steps,
            parameters.hmc_Δτ,
            parameters.hmc_numsmear,
            parameters.hmc_ρstout,
            parameters.hb_MAXIT,
            parameters.hb_numHB,
            parameters.eo,
            parameters.numOR,
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
                meta_enabled = meta_enabled,
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

    function update!(updatemethod::T, U) where {T<:AbstractUpdate}
        error("updatemethod type $(typeof(updatemethod)) is not supported")
    end

end
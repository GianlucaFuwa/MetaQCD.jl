module AbstractUpdateModule
    using LinearAlgebra
    using Random
    using StaticArrays
    using Polyester
    using TimerOutputs
    using Unicode
    using ..Utils
    using ..VerbosePrint

    import ..SystemParameters: Params
    import ..Gaugefields: calc_gauge_action, calc_kinetic_energy, clear!, Gaugefield
    import ..Gaugefields: gaussian_momenta!, Liefield, recalc_gauge_action!, staple
    import ..Gaugefields: staple_eachsite!, substitute_U!, TemporaryField
    import ..Metadynamics: BiasPotential, return_derivative, update_bias!
    import ..AbstractSmearingModule: calc_smearedU, stout_recursion!, StoutSmearing
    import ..AbstractMeasurementModule: top_charge
    import ..UniverseModule: Univ

    abstract type AbstractUpdate end

    include("./hmc.jl")
    include("./metropolis.jl")
    include("./heatbath.jl")
    include("./overrelaxation.jl")
    include("./hbor.jl")
    
    function Updatemethod(parameters::Params, univ::Univ, instance::Int = 1)
        updatemethod = Updatemethod(
            univ.U[instance],
            parameters.update_method,
            parameters.ϵ_metro,
            parameters.multi_hit,
            parameters.metro_target_acc,
            parameters.Δτ,
            parameters.hmc_steps,
            parameters.integrator,
            parameters.MAXIT,
            parameters.numHB,
            parameters.numOR,
            parameters.meta_enabled
        )
            return updatemethod
    end

    function Updatemethod(
        U,
        update_method,
        ϵ_metro = 0.1,
        multi_hit = 1,
        metro_target_acc = 0.5,
        Δτ = 0.1,
        hmc_steps = 10,
        integrator = "leapfrog",
        MAXIT = 1,
        numHB = 1,
        numOR = 4,
        meta_enabled = false,
    )
        if Unicode.normalize(update_method, casefold = true) == "hmc"
            updatemethod = HMCUpdate(
                integrator,
                hmc_steps,
                Δτ,
                U,
                meta_enabled = meta_enabled,
            )
        elseif Unicode.normalize(update_method, casefold = true) == "metropolis"
            updatemethod = MetroUpdate(
                U,
                ϵ_metro,
                multi_hit, 
                metro_target_acc,
                meta_enabled,
            )
        elseif Unicode.normalize(update_method, casefold = true) == "heatbath"
            updatemethod = HBORUpdate(
                U,
                MAXIT, 
                numHB,
                numOR,
            )
        else
            error("update method $(update_method) is not supported")
        end

        return updatemethod
    end

    function update!(Updatemethod::T, U) where {T<:AbstractUpdate}
        error("updatemethod type $(typeof(updatemethod)) is not supported")
    end

end
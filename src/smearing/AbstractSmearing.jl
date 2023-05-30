module AbstractSmearingModule
    using Base.Threads: nthreads, threadid
    using LinearAlgebra
    using Polyester
    using StaticArrays
    using TimerOutputs
    using ..Utils

    import ..Gaugefields: AbstractGaugeAction, CoeffField, Gaugefield
    import ..Gaugefields: staple_eachsite!, substitute_U!, TemporaryField
    
    abstract type AbstractSmearing end

    struct NoSmearing <: AbstractSmearing end

    include("./stout.jl")

    function construct_smearing(U, smearingparameters, coefficient, numlayers)
        if smearingparameters == "nothing"
            smearing = NoSmearing()
        elseif smearingparameters == "stout"
            @assert coefficient !== nothing "Stout coefficient must be set"
            println("Stout smearing will be used")
            smearing = StoutSmearing(U, numlayers, coefficient)
        else
            error("Smearing = $smearing is not supported")
        end

        return smearing
    end

    function calc_smearedU!(smearing, Uin::Gaugefield)
        if smearing !== nothing && typeof(smearing) !== NoSmearing
            apply_smearing!(smearing, Uin)
        end

        return nothing
    end
end
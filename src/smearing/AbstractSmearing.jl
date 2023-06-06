module AbstractSmearingModule
    using Base.Threads: nthreads, threadid, @threads
    using LinearAlgebra
    using Polyester
    using StaticArrays
    using TimerOutputs
    using ..Utils

    import ..Gaugefields: AbstractGaugeAction, CoeffField, Gaugefield, Liefield
    import ..Gaugefields: Temporaryfield, leftmul!, staple_eachsite!, substitute_U!
    
    abstract type AbstractSmearing end

    struct NoSmearing <: AbstractSmearing end

    include("./stout.jl")
    include("./gradientflow.jl")

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

    function calc_smearedU!(smearing, Uin)
        if typeof(smearing) <: StoutSmearing
            apply_smearing!(smearing, Uin)
        elseif typeof(smearing) <: GradientFlow
            flow!(smearing, Uin)
        end

        return nothing
    end

end
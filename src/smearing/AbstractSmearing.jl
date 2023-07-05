"""
    AbstractSmearingModule

Module containing all smearing routines (stout for HMC/MetaD and gradflow for measurements)
Smearing types are subtypes of AbstractSmearing

    StoutSmearing -> struct holding temporary fields for stout smearing and stout force
                     recursion and the stout smearing parameters
                     is parametrized by TG, i.e. the type of Gaugefield and action

    GradientFlow -> struct holding temporary fields for flow and its parameters
                    is parametrized by the integrator, with supported integrators being
                    Euler, RK3 and RK3W7

"""
module AbstractSmearingModule
    using Base.Threads: nthreads, threadid, @threads
    using LinearAlgebra
    using Polyester
    using StaticArrays
    using TimerOutputs
    using ..Utils

    import ..Gaugefields: AbstractGaugeAction, CoeffField, Gaugefield, Liefield,
        Temporaryfield
    import ..Gaugefields: leftmul_dagg!, staple_eachsite!, substitute_U!

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

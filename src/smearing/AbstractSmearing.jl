module AbstractSmearingModule
    #using Base.Threads
    using LinearAlgebra
    using Polyester
    using StaticArrays
    using ..Utils

    import ..Gaugefields: Gaugefield, move, SiteCoords, staple_eachsite!, TemporaryField
    import ..Gaugefields: Liefield
    import ..Metadynamics: BiasPotential, return_derivative

    abstract type AbstractSmearing end

    struct NoSmearing <: AbstractSmearing end

    include("./stout.jl")

    function construct_smearing(smearingparameters, coefficient, numlayers)
        if smearingparameters == "nothing"
            smearing = NoSmearing()
        elseif smearingparameters == "stout"
            @assert coefficient !== nothing "Stout coefficient must be set"
            println("Stout smearing will be used")
            smearing = StoutSmearing(numlayers, coefficient)
        else
            error("Smearing = $smearing is not supported")
        end

        return smearing
    end

    function calc_smearedU(Uin::Gaugefield, smearing)
        if smearing !== nothing && typeof(smearing) !== Nosmearing
            Uout_multi, staples_multi, Qs_multi = apply_smearing(Uin, smearing)
        else
            Uout_multi = nothing
            staples_multi = nothing
            Qs_multi = nothing
        end

        return Uout_multi, staples_multi, Qs_multi
    end
end
module Abstractsmearing_module
    using StaticArrays
    using LinearAlgebra
    using Polyester
    #using Base.Threads

    using ..Utils
    import ..Gaugefields: Gaugefield,Site_coords,Temporary_field,staple_eachsite!,move
    import ..Liefields: Liefield
    import ..Metadynamics: Bias_potential,ReturnDerivative

    abstract type Abstractsmearing end

    struct Nosmearing <: Abstractsmearing end

    include("./stout.jl")

    function construct_smearing(smearingparameters, coefficient, numlayers)
        if smearingparameters == "nothing"
            smearing = Nosmearing()
        elseif smearingparameters == "stout"
            @assert coefficient !== nothing "Stout coefficient must be set"
            println("Stout smearing will be used")
            smearing = Stoutsmearing(numlayers, coefficient)
        else
            error("Smearing = $smearing is not supported")
        end
        return smearing
    end

    function calc_smearedU(Uin::Gaugefield, smearing)
        if smearing !== nothing && typeof(smearing) !== Nosmearing
            Uout_multi, staples_multi, Qs_multi = apply_smearing_U(Uin, smearing)
        else
            Uout_multi = nothing
            staples_multi = nothing
            Qs_multi = nothing
        end
        return Uout_multi, staples_multi, Qs_multi
    end

end



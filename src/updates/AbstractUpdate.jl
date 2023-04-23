module AbstractUpdate_module
    using Random
    using StaticArrays
    using LinearAlgebra
    using Polyester
    using ..Utils
    #using Base.Threads
    using LoopVectorization

    import ..System_parameters: Params
    import ..Verbose_print: Verbose_level,println_verbose2,println_verbose3
    import ..Utils: exp_iQ,gen_SU3_matrix
    import ..Gaugefields: Gaugefield,Temporary_field,Site_coords,
        staple_eachsite!,calc_GaugeAction,recalc_GaugeAction!,staple,
        get_Sg,get_CV,get_β,set_Sg!,set_CV!,
        substitute_U!
    import ..Liefields: Liefield,gaussianP!,trP2,clear_P!
    import ..Metadynamics: Bias_potential,
        update_bias!,ReturnDerivative,DeltaV,
        get_smearparams_for_CV,get_kind_of_CV
    import ..Abstractsmearing_module: Stoutsmearing,calc_smearedU,stout_recursion!
    import ..AbstractMeasurement_module: top_charge
    import ..Universe_module: Univ

    abstract type AbstractUpdate end

    include("./hmc.jl")
    include("./localmc.jl")
    include("./heatbath.jl")
    include("./overrelaxation.jl")
    include("./hbor.jl")
    
    function Updatemethod(parameters::Params, univ::Univ, instance=1)
        updatemethod = Updatemethod(
            univ.U[instance],
            parameters.update_method,
            parameters.ϵ_metro,
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
        Δτ = 0.1,
        hmc_steps = 10,
        integrator = "Leapfrog",
        MAXIT = 1,
        numHB = 4,
        numOR = 1,
        meta_enabled = false,
        )
        if update_method == "HMC"
            updatemethod = HMC_update(
                integrator,
                hmc_steps,
                Δτ,
                U,
                meta_enabled = meta_enabled,
            )
        elseif update_method == "Local"
            updatemethod = Local_update(
                ϵ_metro,
                meta_enabled,
            )
        elseif update_method == "Heatbath"
            updatemethod = HBOR_update(
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
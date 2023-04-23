module Mainbuild
    using DelimitedFiles
    using InteractiveUtils
    using Dates
    
    import ..System_parameters: Params,Params_set,parameterloading
    import ..Verbose_print: Verbose_,println_verbose
    import ..Gaugefields: Gaugefield,recalc_CV!,RandomGauges!,IdentityGauges!,normalize!,substitute_U!
    import ..Liefields: Liefield
    import ..Metadynamics: Bias_potential
    import ..Measurements: Measurement_set,build_measurements
    import ..LocalMC: loc_metro_sweep!,loc_metro_sweep_meta!
    import ..HMC: HMC!

    import ..System_parameters:physical,meta,sim,mc,meas,system

    function run_build(filenamein::String)
        filename = filenamein
        include(abspath(filename))
        params_set = Params_set(physical,meta,sim,mc,meas,system)

        run_build(params_set)

        return nothing
    end

    function run_build(params_set::Params_set)
        params = parameterloading(params_set)
        gfield = Gaugefield(params)
        gfieldcopy1 = deepcopy(gfield)
        gfieldcopy2 = deepcopy(gfield)
        pfield = Liefield(gfield.NX,gfield.NY,gfield.NZ,gfield.NT)
        bias = Bias_potential(params)
        run_build!(gfield,gfieldcopy1,gfieldcopy2,pfield,bias,params)

        return nothing
    end

    function run_build!(gfield::Gaugefield,gfieldcopy1::Gaugefield,gfieldcopy2::Gaugefield,pfield::Liefield,bias::Bias_potential,params::Params)
        verbose = Verbose_(params.logfile)
        println_verbose(verbose,"# ",pwd())
        println_verbose(verbose,"# ",Dates.now())
        versioninfo(verbose)
        println_verbose(verbose,"- - - - - - - - - - - - - - - - - - -")

        measset = Measurement_set(params.measure_dir)
        rng = params.randomseeds[1]

        if params.initial == "hot"
            RandomGauges!(gfield,rng)
        elseif params.initial == "cold"
            IdentityGauges!(gfield) 
        end

        recalc_CV!(gfield)

        println_verbose(verbose,"# Thermalization Start at ",Dates.now(),"\n")
        if params.update_method == "Local" || params.update_method == "LocalMeta"
            for i = 1:params.Ntherm
                loc_metro_sweep!(gfield,rng,params.ϵ_metro)
                #normalize!(gfield)
            end
        elseif params.update_method == "HMC" || params.update_method == "HMCMeta"
            for i = 1:params.Ntherm
                HMC!(gfield,pfield,params.ϵ_hmc,params.hmc_steps,rng)
                #normalize!(gfield)
            end
        end
        println_verbose(verbose,"# Thermalization Done at ",Dates.now(),"\n")

        numaccepts = 0
        for trj = 1:params.Nsweeps
            if params.update_method == "Local"
                tmp = loc_metro_sweep!(gfield,rng,params.ϵ_metro)
            elseif params.update_method == "Local-Meta"
                tmp = loc_metro_sweep_meta!(gfield,bias,rng,params.ϵ_metro,false)
            elseif params.update_method == "HMC"
                tmp = HMC!(gfield,pfield,params.ϵ_hmc,params.hmc_steps,rng)
            elseif params.update_method == "HMC-Meta"
                tmp = HMC_meta!(gfield,pfield,bias,params.ϵ_hmc,params.hmc_steps,false)
            end
            numaccepts += tmp
            #normalize!(gfield)
            if trj%params.measure_every == 0
                substitute_U!(gfieldcopy1,gfield)
                substitute_U!(gfieldcopy2,gfield)
                build_measurements(trj,gfield,gfieldcopy1,gfieldcopy2,measset,params.numsmear)
            end
        end
        #=
        open(params.biasfile,"w") do io
            writedlm(io,[bias.q_vals bias.values])
        end    
        =#
        if params.update_method == "Local"
            println_verbose(verbose,"# Local Acceptance rate: ",100*numaccepts/params.Nsweeps/gfield.NV/4,"%")
        elseif params.update_method == "HMC"
            println_verbose(verbose,"# HMC Acceptance rate: ",100*numaccepts/params.Nsweeps,"%")
        end
        println_verbose(verbose,"- - - - - - - - - - - - - - - - - - -")

        #println_verbose(verbose,"Metapotential has been saved in file \"$(params.biasfile)\"")
        flush(stdout)
        flush(verbose)
        close(verbose.fp)
        return nothing
    end

end

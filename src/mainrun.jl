module Mainrun
    using Printf 
    using DelimitedFiles
    using InteractiveUtils
    using Dates
    using Distributed
    using Base.Threads:@spawn,nthreads
    
    import ..System_parameters:Params,Params_set,parameterloading
    import ..Verbose_print:Verbose_,println_verbose
    import ..Gaugefields:Gaugefield,recalc_CV!,RandomGauges!,IdentityGauges!,normalize!
    import ..Liefields:Liefield
    import ..Metadynamics:Bias_potential
    import ..Measurements:Measurement_set,measurements,calc_weights,build_measurements
    import ..LocalMC:PT_swap!,loc_metro_sweep!,loc_metro_sweep_meta!
    import ..HMC:HMC!

    import ..System_parameters:physical,meta,sim,mc,meas,system

    function run_sim(filenamein::String)
        filename = filenamein
        include(abspath(filename))
        params_set = Params_set(physical,meta,sim,mc,meas,system)

        run_sim(params_set)

        return nothing
    end

    function run_sim(params_set::Params_set)
        params = parameterloading(params_set)

        if params.meta_enabled
            if params.tempering_enabled
                @assert nthreads() ≥ 2 "Make sure to enable at least 2 threads when using parallel tempering! Do: 'julia -t 2' or 'julia -t auto'"
                gfield_main = Gaugefield(params)
                gfield_meta = Gaugefield(params)
                pfield_main = Liefield(gfield.NX,gfield.NY,gfield.NZ,gfield.NT)
                pfield_meta = Liefield(gfield.NX,gfield.NY,gfield.NZ,gfield.NT)
                bias = Bias_potential(params)
                run_temperedsim!(gfield_main,gfield_meta,pfield_main,pfield_meta,bias,params)
            elseif ~params.parallel_tempering
                gfield = Gaugefield(params)
                pfield = Liefield(gfield.NX,gfield.NY,gfield.NZ,gfield.NT)
                bias = Bias_potential(params)
                run_sim!(gfield,pfield,bias,params)
            end
        else
            gfield = Gaugefield(params)
            pfield = Liefield(gfield.NX,gfield.NY,gfield.NZ,gfield.NT)
            run_sim!(gfield,pfield,params)
        end

        return nothing
    end

    function run_sim!(gfield::Gaugefield,pfield::Liefield,bias::Bias_potential,params::Params)
        verbose = Verbose_(params.logfile)
        println_verbose(verbose,"# ",pwd())
        println_verbose(verbose,"# ",Dates.now())
        versioninfo(verbose)

        measset = Measurement_set(params.measure_dir,meas_calls = params.meas_calls)
        rng = params.randomseeds[1]

        if params.initial == "hot"
            RandomGauges!(gfield,rng)
        elseif params.initial == "cold"
            IdentityGauges!(gfield) 
        end

        recalc_Q!(gfield)

        for i = 1:params.Ntherm
            if params.update_method == "Local" || params.update_method == "Local-Meta"
                loc_metro_sweep!(gfield,rng,params.ϵ_metro)
            elseif params.update_method == "HMC"
                HMC!(gfield,pfield,params.ϵ_hmc,params.hmc_steps,rng,verbose)
            end
            #normalize!(gfield)
        end
        println("Thermalization Done \n")
        
        numaccepts = 0
        bias_mean = deepcopy(bias.values)
        for trj = 1:params.Nsweeps
            if params.update_method == "Local"
                tmp = loc_metro_sweep!(gfield,rng,params.ϵ_metro)
            elseif params.update_method == "Local-Meta"
                tmp = loc_metro_sweep_meta!(gfield,bias,rng,params.ϵ_metro,false,verbose)
            elseif params.update_method == "HMC"
                tmp = HMC!(gfield,pfield,params.ϵ_hmc,params.hmc_steps,rng)
            elseif params.update_method == "HMC-Meta"
                tmp = HMC_meta!(gfield,pfield,bias,params.ϵ_hmc,params.hmc_steps,false,verbose)
            end
            numaccepts += tmp
            bias_mean += bias.values
            measurements(trj,field,measset)
        end
        bias.values = bias_mean ./ (params.Nsweeps÷2)
        #=
        open(params.biasfile,"w") do io
            writedlm(io,[bias.q_vals bias.values])
        end    
        =#
        if params.update_method == "Local" || params.update_method == "Local-Meta"
            println_verbose(verbose,"Acceptance rate: $(100*numaccepts/params.Nsweeps/gfield.NX/gfield.NY/gfield.NZ/gfield.NT/4)%")
        elseif params.update_method == "HMC" || params.update_method == "HMC-Meta"
            println_verbose(verbose,"Acceptance rate: $(100*numaccepts/params.Nsweeps)%")
        end
        println_verbose(verbose,"- - - - - - - - - - - - - - - - - - -")

        q_vals = readdlm(pwd()*"/"*params.measure_dir*"/Continuous_charge.txt",Float64,comments=true)
        weights = calc_weights(q_vals[:,2],bias)
        open(params.weightfile,"w") do io
            writedlm(io,weights)
        end
        println_verbose(verbose,"Weights have been saved in file \"$(params.weightfile)\"")

        flush(stdout)
        flush(verbose)
        return nothing
    end

    function run_sim!(gfield::Gaugefield,pfield::Liefield,params::Params)
        verbose = Verbose_(params.logfile)
        println_verbose(verbose,"# ",pwd())
        println_verbose(verbose,"# ",Dates.now())
        versioninfo(verbose)

        measset = Measurement_set(params.measure_dir,meas_calls = params.meas_calls)
        rng = params.randomseeds[1]

        if params.initial == "hot"
            RandomGauges!(gfield,rng)
        elseif params.initial == "cold"
            IdentityGauges!(gfield) 
        end

        recalc_Q!(gfield)

        println_verbose(verbose,"# Thermalization Start at ",Dates.now(),"\n")
        if params.update_method == "Local" || params.update_method == "LocalMeta"
            for i = 1:params.Ntherm
                loc_metro_sweep!(gfield,rng[1],params.ϵ_metro)
                #normalize!(gfield)
            end
        elseif params.update_method == "HMC" || params.update_method == "HMCMeta"
            for i = 1:params.Ntherm
                HMC!(gfield,pfield,params.ϵ_hmc,params.hmc_steps,false,rng[1],verbose)
                #normalize!(gfield)
            end
        end
        println_verbose(verbose,"# Thermalization Done at ",Dates.now(),"\n")
        
        numaccepts = 0
        for trj = 1:params.Nsweeps
            if params.update_method == "Local"
                tmp = loc_metro_sweep!(gfield,rng,params.ϵ_metro)
            elseif params.update_method == "HMC"
                tmp = HMC!(gfield,pfield,params.ϵ_hmc,params.hmc_steps,rng,verbose)
            end
            numaccepts += tmp
            measurements(trj,field,measset)
        end

        if params.update_method == "Local" || params.update_method == "Local-Meta"
            println_verbose(verbose,"Acceptance rate: $(100*numaccepts/params.Nsweeps/gfield.NX/gfield.NY/gfield.NZ/gfield.NT/4)%")
        elseif params.update_method == "HMC" || params.update_method == "HMC-Meta"
            println_verbose(verbose,"Acceptance rate: $(100*numaccepts/params.Nsweeps)%")
        end
        println_verbose(verbose,"- - - - - - - - - - - - - - - - - - -")

        flush(stdout)
        flush(verbose)
        return nothing
    end

    function run_temperedsim!(gfields::Vector{Gaugefield},pfields::Vector{Liefield},biases::Vector{Bias_potential},params::Params)
        Ninstances = params.Ninstances
        verbose = Verbose_(params.logfile)
        println_verbose(verbose,"# ",pwd())
        println_verbose(verbose,"# ",Dates.now())
        versioninfo(verbose)
        println_verbose(verbose,"# Parallel tempered run with ",params.Ninstances," instances")

        meassets = []
        for i=1:params.Ninstances
            push!(meassets,Measurement_set(params.measure_dir[i],meas_calls = params.meas_calls[i])) 
        end

        ϵ = params.ϵ_metro
        rng = params.randomseeds

        if params.initial == "hot"
            for (idx,gfield) in enumerate(gfields)
            RandomGauges!(gfield,rng[idx])
            recalc_Q!(gfield)
            end
        elseif params.initial == "cold"
            for (idx,gfield) in enumerate(gfields)
            IdentityGauges!(gfield,rng[idx])
            recalc_Q!(gfield)
            end
        end

        println("Thermalization Begin")
        for i = 1:params.Ntherm
            if params.update_method == "Local" 
                for (idx,gfield) in enumerate(gfields)
                loc_metro_sweep!(gfield,rng[idx],params.ϵ_metro)
                end
            elseif params.update_method == "HMC" 
                for (idx,gfield) in enumerate(gfields)
                HMC!(gfield,pfields[idx],params.ϵ_hmc,params.hmc_steps,false,rng[idx],verbose)
                end
            end
            #normalize!(gfield)
        end
        println("Thermalization Done")

        numaccepts = zeros(Int64,Ninstances)
        num_swaps = zeros(Int64,Ninstances-1)
        bias_means = []
        for (idx,bias) in enumerate(biases)
            push!(bias_means,bias.values)
        end

        for trj = 1:params.Nsweeps
            if params.update_method == "Local" 
                numaccepts[1] += loc_metro_sweep!(gfields[1],rng[1],params.ϵ_metro)
                for i=2:Ninstances
                numaccepts[i] += loc_metro_sweep_meta!(gfields[i],biases[i],rng[i],params.ϵ_metro,true)
                end
            elseif params.update_method == "HMC" 
                numaccepts[1] += HMC!(gfields[1],pfields[1],params.ϵ_hmc,params.hmc_steps,true,rng[1],verbose)
                for i=2:Ninstances
                numaccepts[i] += MC_meta!(gfields[i],pfields[i],biases[i],params.ϵ_hmc,params.hmc_steps,true,rng[i],verbose)
                end
            end
            for (idx,bias) in enumerate(biases)
                bias_means[idx] += bias.values
            end
            if trj%params.swap_every == 0
                for i=1:Ninstances-1
                accept_swap = try_swap!(gfields[i],gfields[i+1],biases[i],biases[i+1],rng[1],false)
                num_swaps[i] += ifelse(accept_swap,1,0)
                end
            end
            for (idx,gfield) in enumerate(gfields)
            measurements(trj,gfield,gfieldcopies1[idx],gfieldcopies2[idx],meassets[idx])
            end
        end

        for i=Ninstances
        println_verbose(verbose,"Main Acceptance rate: ",100*numaccepts_main/params.Nsweeps/gfields[1].NV,"%")
        println_verbose(verbose,"Meta Acceptance rate: ",100*numaccepts_meta/params.Nsweeps/gfields[1].NV,"%")
        println_verbose(verbose,"Swap Acceptance rate: ",100*num_swaps/(params.Nsweeps÷params.swap_every),"%")
        end
        #=
        bias.values = bias_mean ./ params.Nsweeps
        open(params.biasfile,"w") do io
            writedlm(io,[bias.q_vals bias.values])
        end
        println_verbose(verbose,"Metapotential has been saved in file \"$(params.biasfile)\"")
        q_vals = readdlm(pwd()*"/"*params.measure_dir_secondary*"/Continuous_charge.txt",Float64,comments=true)
        weights = calc_weights(q_vals[:,2],bias)
        open(params.weightfile,"w") do io
            writedlm(io,weights)
        end
        println_verbose(verbose,"Weights have been saved in file \"$(params.weightfile)\"")
        =#
        flush(stdout)
        flush(verbose)
        return nothing
    end

end

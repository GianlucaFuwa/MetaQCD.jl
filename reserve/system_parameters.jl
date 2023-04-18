module System_parameters
	using Random
    using DelimitedFiles

    export Params
	
	printlist_physical = ["L","β","NC"]

	printlist_meta = ["meta_enabled","symmetric","CVlims","bin_width","w","k","is_static"]

	printlist_sim = ["Ntherm","Nsweeps","initial","tempering_enabled","numinstances","swap_every"]

    printlist_mc = ["update_method","integrator","ϵ_local","ϵ_hmc","hmc_steps"]

    printlist_meas = ["meas_calls","measure_every","smearing_type","numsmear","ρ_smear"]

    printlist_system = ["veryverbose","randomseeds","logdir","logfile","loadfile",
    "measure_dir","savebias_dir","biasfile","usebias","weightfile"]

    const printlists = [printlist_physical,printlist_meta,printlist_sim,printlist_mc,printlist_meas,printlist_system]
    const printlists_header = ["# Physical Settings ","# Metadynamics Settings","# Simulation Settings",
    "# MC Settings","# Measurement Settings","# System Settings"]

	defaultmeasures = Array{Dict,1}(undef,2)
	for i=1:length(defaultmeasures)
		defaultmeasures[i] = Dict()
	end
	defaultmeasures[1]["methodname"] = "Topological_charge"
    defaultmeasures[2]["methodname"] = "Action"

	physical = Dict()
	meta = Dict()
	sim = Dict()
    mc = Dict()
    meas = Dict()
	system = Dict()

    meta["meta_enabled"] = false
    meta["is_static"] = [false]

    mc["update_method"] = "HMC"
    mc["integrator"] = "Leapfrog"
    mc["ϵ_hmc"] = 0.02
    mc["hmc_steps"] = 50

    sim["tempering_enabled"] = false

    meas["meas_calls"] = defaultmeasures
    meas["measure_every"] = 1

    system["randomseeds"] = [Random.Xoshiro(1206)]
	
	mutable struct Params_set
	physical::Dict
	meta::Dict
	sim::Dict
    mc::Dict
    meas::Dict
	system::Dict
	
        function Params_set(physical,meta,sim,mc,meas,system)
            return new(physical,meta,sim,mc,meas,system)
        end
    end
	
    function make_parameters(physical,meta,sim,mc,meas,system)
        return Params_set(physical,meta,sim,mc,meas,system)
    end
    
	struct Params
		L::NTuple{4,Int64}
		β::Float64
        NC::Int64

        meta_enabled::Bool
        symmetric::Union{Nothing,Bool}
		CVlims::Union{Nothing,NTuple{2,Float64}}
		bin_width::Union{Nothing,Float64}
		w::Union{Nothing,Float64}
		k::Union{Nothing,Float64}
        is_static::Union{Nothing,Vector{Bool}}

        Ntherm::Int64
		Nsweeps::Int64
		initial::String
        tempering_enabled::Union{Nothing,Bool}
        numinstances::Union{Nothing,Int64}
        swap_every::Union{Nothing,Int64}

        update_method::String
		ϵ_local::Union{Nothing,Float64}
        integrator::Union{Nothing,String}
        ϵ_hmc::Union{Nothing,Float64}
        hmc_steps::Union{Nothing,Int64}

        meas_calls::Array{Dict,1}
        measure_every::Int64
        smearing_type::Union{Nothing,String}
        numsmear::Union{Nothing,Int64}
        ρ_smear::Union{Nothing,Float64}

		randomseeds::Vector{Xoshiro}
		logdir::String
		logfile::String
		loadfile::IOStream
		measurement_dir::String
        measurement_dir_secondary::Union{Nothing,String}
		savebias_dir::Union{Nothing,String}
		biasfile::Union{Nothing,Vector{String}}
		usebias::Union{Nothing,Vector{Union{Nothing,String}}}
        weightfile::Union{Nothing,Vector{String}}

		function Params(physical,meta,sim,mc,meas,system)
			L = physical["L"]
			β = physical["β"]
            NC = 3
            ### Setup Metadynamics ###
            meta_enabled = meta["meta_enabled"]
            if meta_enabled
                symmetric = meta["symmetric"]
                CVlims = meta["CVlims"]
                bin_width = meta["bin_width"]
                w = meta["w"]
                k = meta["k"]
                is_static = meta["is_static"]
                savebias_dir = system["savebias_dir"]
                biasfile = []
                usebias = []
                weightfile = []
                tempering_enabled = sim["tempering_enabled"]
                if tempering_enabled
                    swap_every = sim["swap_every"]
                    numinstances = sim["numinstances"]
                    if haskey(system,"usebias")
                        usebias = system["usebias"]
                    else
                        for i=1:numinstances
                        push!(usebias,nothing)
                        end
                    end
                    for i=1:numinstances
                        push!(biasfile,pwd()*savebias_dir*"/"*system["biasfile"]*"_$i.txt")
                        push!(weightfile,pwd()*"/"*measure_dir*"Weights_$i.txt") 
                    end
                else # IF NO TEMPERING
                    push!(biasfile,pwd()*savebias_dir*"/"*system["biasfile"]*".txt")
                    push!(weightfile,pwd()*"/"*measure_dir*"Weights.txt") 
                end # END IF TEMPERING
                if isdir(savebias_dir) == false
                    mkpath(savebias_dir)
                end
                if haskey(system,"usebias")
                    push!(usebias,system["usebias"][1])
                else
                    push!(usebias,nothing)
                end
            else # IF NO META
                symmetric = nothing
                CVlims = nothing
                bin_width = nothing
                w = nothing
                k = nothing
                is_static = nothing
                savebias_dir = nothing
                biasfile = nothing
                usebias = nothing
                weightfile = nothing
                tempering_enabled = nothing
            end # END IF META
			########################

			Ntherm = sim["Ntherm"]
			Nsweeps = sim["Nsweeps"]
            initial = sim["initial"]

            update_method = mc["update_method"]
            if update_method == "HMC" || update_method == "HMC-Meta"
                integrator = mc["integrator"]
                ϵ_hmc = mc["ϵ_hmc"]
                hmc_steps = mc["hmc_steps"]
                ϵ_local = nothing
            elseif  update_method == "Local" || update_method == "Local-Meta"
                integrator  = nothing
                ϵ_hmc = nothing
                hmc_steps = nothing
                ϵ_local = mc["ϵ_local"]
            else
                error("Update method not supported - Only 'Local' and 'HMC' or 'Local-Meta' and 'HMC-Meta'")
            end

			meas_calls = meas["meas_calls"]
            measure_every = meas["measure_every"]
            if haskey(meas,"smearing_type")
                if smearing_type == "stout"
                    ρ_smear = meas["ρ_stout"]
                else
                    error("Smearing type not supported - Only 'stout'")
                end
                smearing_type = meas["smearing_type"]
                numsmear = meas["numsmear"]
            else
                smearing_type = nothing
                numsmear = nothing
                ρ_smear = nothing
            end

            veryverbose = system["veryverbose"]
            randomseeds = system["randomseeds"]
			logdir = system["logdir"]
			if isdir(logdir) == false
				mkdir(logdir)
			end
			logfile = pwd()*"/"*logdir*"/"*system["logfile"]
			loadfile = open(logfile,"a")

			measure_dir = system["measure_dir"]
			if isdir(measure_dir) == false
				mkpath(measure_dir)
            end

			return new(
				L,β,NC,
                meta_enabled,symmetric,CVlims,bin_width,w,k,is_static,
                Ntherm,Nsweeps,initial,tempering_enabled,numinstances,swap_every,
                update_method,ϵ_local,integrator,ϵ_hmc,hmc_steps,
                meas_calls,measure_every,smearing_type,numsmear,ρ_smear,
                veryverbose,randomseeds,logdir,logfile,loadfile,measure_dir,savebias_dir,biasfile,usebias,weightfile)
		end

		function Params(params_set::Params_set)
			return Params(params_set.physical,params_set.meta,params_set.sim,params_set.mc,params_set.meas,params_set.system)
		end
	end
    
    function set_params(dict,string)
        if haskey(dict,string)
            return dict[string]
        else
            error("No parameters given. \n")
        end
    end
	
	function make_parametersdict(p::T) where T<:Any
		pnames = fieldnames(T)
		pdict = Dict()
		for i in eachindex(pnames)
			pdict[String(pnames[i])] = getfield(p,pnames[i])
		end
		return pdict,pnames
	end

	function print_parameters_file(p)
		filename = p.logfile*"_parameters.jl"
		fp = open(filename,"w")
		println(fp,"# - - parameters - - - - - - - - ")
		pdict, = make_parametersdict(p)
		for param in pdict
			if typeof(param[2]) == String
				println(fp,"$(param[1]) = \"$(param[2])\"")
			else
				println(fp,"$(param[1]) = $(param[2])")
			end
		end
		println(fp,"# - - - - - - - - - - - - - - - -")
		close(fp)
	end
	
    function setprint(fp,fp2,string)
        println(fp,string)
        if fp2 !== nothing
            println(fp2,string)
        end
        println(string)
    end
    
    function get_stringfromkey(key)
        if typeof(key) == String
            string = "\"$key\""
        else
            string = "$key"
		end
	end

	function get_header(params_set::Params_set,inputname)
		if haskey(params_set.system,inputname)
            return "system",params_set.system[inputname]
        elseif haskey(params_set.physical,inputname)
            return "physical",params_set.physical[inputname]
        elseif haskey(params_set.meta,inputname)
            return "meta",params_set.meta[inputname]
        elseif haskey(params_set.mc,inputname)
            return "mc",params_set.mc[inputname]
        elseif haskey(params_set.meas,inputname)
            return "meas",params_set.meas[inputname]
        elseif haskey(params_set.sim,inputname)
            return "sim",params_set.sim[inputname]
        else
            return nothing,nothing
        end
    end

	function print_measurementinfo(fp,fp2,key)
        string = "meas[\"meas_calls\"] = Dict[ "
        setprint(fp,fp2,string)
        for (i,data) in enumerate(key)
            string = "  Dict{Any,Any}(\"methodname\" => \"$(data["methodname"])\","
            setprint(fp,fp2,string)
            string = "  )"
            if i != length(key)
                string *= ","
            end 
            setprint(fp,fp2,string)
        end
        string = "]"
        setprint(fp,fp2,string)
    end

    function print_parameters_list(params_set::Params_set,p=nothing;filename=nothing)
        if filename === nothing 
            @assert p !== nothing "wrong input!"

            filename = p.logdir*"/parameters.jl"
            fp2 = p.loadfile
        else
            fp2 = nothing
        end
        fp = open(filename,"w")
        
        setprint(fp,fp2,"# - - parameters - - - - - - - - - - - ")
        for (i,printlist_i) in enumerate(printlists)
            setprint(fp,fp2,printlists_header[i] )
            for name in printlist_i
                headstring,key = get_header(params_set,name)
                if headstring !== nothing
                    if name == "meas_calls"
                        print_measurementinfo(fp,fp2,key)
                    else
                        string = get_stringfromkey(key)
                        paramstring = headstring*"[\"$name\"] = "*string
                        setprint(fp,fp2,paramstring)
                    end
                end
            end
            setprint(fp,fp2,"\t" )
        end
 
        setprint(fp,fp2,"# - - - - - - - - - - - - - - - - - - -")

        close(fp)
        close(fp2)

        println("""
        # Your parameters were written in $filename
        # If you want to do the simulation with same parameters, 
        # Just do 
        # julia run.jl $filename
        """)
        flush(stdout)
        return nothing
    end

    function print_parameters(params_set::Params_set,p)
        print_parameters_list(params_set,p)
        return nothing
    end

    function print_parameters(filename,params_set::Params_set)
        print_parameters_list(params_set,filename=filename)
		return nothing
    end


    function print_parameters(p)
        println("# - - parameters - - - - - - - - - - - ")
        
        pdict,pnames = make_parametersdict(p)
        for param in pdict
            if typeof(param[2]) == String
                println("$(param[1]) = \"$(param[2])\"")
            else
                println("$(param[1]) = $(param[2])")
            end
        end
        println("# - - - - - - - - - - - - - - - - - - -")
        
        print_parameters_file(p)
        println("""
        # Your parameters were written in parameters_used.jl
        # If you want to do the simulation with same parameters, 
        # Just do 
        # julia run.jl parameters_used.jl
        """)
        flush(stdout)
        return nothing
    end

    function parameterloading(physical,meta,sim,mc,meas,system)
        param_set = Params_set(physical,meta,sim,mc,meas,system)
        p = Params(param_set)

        print_parameters(param_set,p)
        return p
    end

    function parameterloading(param_set::Params_set)
        p = Params(param_set)

        print_parameters(param_set,p)
        return p
    end
    
end
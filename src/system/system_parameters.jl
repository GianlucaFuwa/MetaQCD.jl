module System_parameters
	using Random
    using DelimitedFiles

    export Params
	
	printlist_physical = ["L","β","NC"]

	printlist_meta = ["meta_enabled","Qmax","Qthr","δq","w","k"]

	printlist_sim = ["Ntherm","Nsweeps","initial","tempering_enabled","swap_every"]

    printlist_mc = ["update_method","ϵ_metro","ϵ_hmc","hmc_steps"]

    printlist_meas = ["meas_calls","measure_every","smearing_type","numsmear","ρ_smear"]

    printlist_system = ["randomseeds","logdir","logfile","loadfile",
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

    mc["update_method"] = "HMC"
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
		L::Tuple
		β::Float64
        NC::Int64

        meta_enabled::Bool
		Qmax::Union{Nothing,NTuple{2,Float64}}
		Qthr::Union{Nothing,NTuple{2,Float64}}
		δq::Union{Nothing,Float64}
		w::Union{Nothing,Float64}
		k::Union{Nothing,Float64}

        Ntherm::Int64
		Nsweeps::Int64
		initial::String
        tempering_enabled::Union{Nothing,Bool}
        swap_every::Union{Nothing,Int64}

        update_method::String
		ϵ_local::Union{Nothing,Float64}
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
		measure_dir::String
        measure_dir_secondary::Union{Nothing,String}
		savebias_dir::Union{Nothing,String}
		biasfile::Union{Nothing,String}
		usebias::Union{Nothing,Array{Float64,1}}
        weightfile::Union{Nothing,String}

		function Params(physical,meta,sim,mc,meas,system)
			L = physical["L"]
			β = physical["β"]
            NC = 3

            meta_enabled = meta["meta_enabled"]
            if meta_enabled
                Qmax = meta["Qmax"]
                Qthr = meta["Qthr"]
                δq = meta["δq"]
                w = meta["w"]
                k = meta["k"]
            else
                Qmax = nothing
                Qthr = nothing
                δq = nothing
                w = nothing
                k = nothing
            end
			
			Ntherm = sim["Ntherm"]
			Nsweeps = sim["Nsweeps"]
            initial = sim["initial"]

            update_method = mc["update_method"]
            if ~meta_enabled
                @assert update_method !== "Local-Meta" || update_method !== "HMC-Meta"
            end
            if update_method == "HMC" || update_method == "HMC-Meta"
                ϵ_hmc = mc["ϵ_hmc"]
                hmc_steps = mc["hmc_steps"]
                ϵ_metro = nothing
            elseif  update_method == "Local" || update_method == "Local-Meta"
                ϵ_hmc = nothing
                hmc_steps = nothing
                ϵ_metro = mc["ϵ_metro"]
            else
                error("Update method not supported - Only 'Local' and 'HMC'")
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

            randomseeds = system["randomseeds"]
			logdir = system["logdir"]
			if isdir(logdir) == false
				mkdir(logdir)
			end
			logfile = pwd()*"/"*logdir*"/"*system["logfile"]
			loadfile = open(logfile,"a")
			measure_dir = system["measure_dir"]

            if meta_enabled
                tempering_enabled = sim["tempering_enabled"]
                if tempering_enabled
                    swap_every = sim["swap_every"]
                    measure_dir_secondary = system["measure_dir"]*"_secondary"
                    weightfile = pwd()*"/"*measure_dir_secondary*"/Weights.txt"
                    if isdir(measure_dir_secondary) == false && measure_dir_secondary !== nothing
                        mkpath(measure_dir_secondary)
                    end
                else 
                    swap_every = nothing
                    measure_dir_secondary = nothing
                    weightfile = pwd()*"/"*measure_dir*"/Weights.txt"
                end
                if haskey(system,"usebias")
                    file = system["usebias"]
                    usebias = readdlm(file,Float64)
                    @assert length(usebias[:,2]) == round(Int,((Qmax[2]-Qmax[1])/δq),RoundNearestTiesAway)+1 "Length of given Metapotential doesn't match Meta-parameters"
                    usebias = usebias[:,2]
                else 
                    usebias = zeros(round(Int,(Qmax[2]-Qmax[1])/δq,RoundNearestTiesAway)+1)
                end
                savebias_dir = system["savebias_dir"]
                if isdir(savebias_dir) == false && savebias_dir !== nothing
                    mkpath(savebias_dir)
                    biasfile = pwd()*"/"*savebias_dir*"/"*system["biasfile"]
                end
            else
                tempering_enabled = nothing
                swap_every = nothing
                measure_dir_secondary = nothing
                weightfile = nothing
                usebias = nothing
                savebias_dir = nothing
                biasfile = nothing
            end

			if isdir(measure_dir) == false
				mkpath(measure_dir)
            end

			return new(
				L,β,NC,meta_enabled,Qmax,Qthr,δq,w,k,Ntherm,Nsweeps,initial,tempering_enabled,swap_every,update_method,ϵ_metro,
                ϵ_hmc,hmc_steps,meas_calls,measure_every,smearing_type,numsmear,ρ_smear,randomseeds,logdir,
                logfile,loadfile,measure_dir,measure_dir_secondary,savebias_dir,biasfile,usebias,weightfile)
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
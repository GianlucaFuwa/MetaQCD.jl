struct HMCLevel{NC,TI,TF,TFP}
    integrator::TI
    numsteps::Int64
    Δτ::Float64
    numchildren::NC
    forces::TF # which forces contribute to this level? INFO: Had to make this a tuple of Vals because printf needs to know
    forcefile::TFP
    function HMCLevel(
        integrator::TI,
        numsteps,
        Δτ,
        forces;
        numchildren::NC=Val(0),
        hmc_logging=true,
        logdir="",
        instance=mpi_myrank(),
        numcv=0,
        distributed=false,
    ) where {NC,TI<:AbstractIntegrator}
        comm_instance = mpi_comm_instance()
        ilevel = _unwrap_val(numchildren) + 1

        forces = ntuple(length(forces)) do i
            Val(forces[i])
        end

        if hmc_logging && (logdir != "") && (!distributed || mpi_amroot(comm_instance))
            for ii in instance
                ext = "$(lpad(ii, 3, "0")).txt"
                _forcefile = joinpath(logdir, "hmc_force_logs_level$(ilevel)_$(ext)")
                force_fp = fopen(_forcefile, "w")

                if Val(0) ∈ forces
                    for icv in 1:numcv
                        printf(force_fp, "%-25s", "avg||F_V$(icv)||")
                        printf(force_fp, "%-25s", "sup||F_V$(icv)||")
                    end
                end

                if Val(1) ∈ forces
                    printf(force_fp, "%-25s", "avg||F_Sg||")
                    printf(force_fp, "%-25s", "sup||F_Sg||")
                end

                for i in _unwrap_val.(forces)
                    i ∈ (0, 1) && continue
                    printf(force_fp, "%-25s", "avg||F_Sf$(i-1)||")
                    printf(force_fp, "%-25s", "sup||F_Sf$(i-1)||")
                end

                newline(force_fp)
                fclose(force_fp)
            end

            ext = "$(lpad(instance[1], 3, "0")).txt"
            _forcefile = joinpath(logdir, "hmc_force_logs_level$(ilevel)_$(ext)")
            forcefile = SStaticString(_forcefile)
        else
            forcefile = nothing
        end

        TF = typeof(forces)
        TFP = typeof(forcefile)
        return new{NC,TI,TF,TFP}(integrator, numsteps, Δτ, numchildren, forces, forcefile)
    end
end

function Base.show(io::IO, ::MIME"text/plain", level::HMCLevel)
    str = """
        |  HMCLevel(
        |    integrator: $(level.integrator)
        |    numsteps: $(level.numsteps)
        |    Δτ: $(level.Δτ)
        |    numchildren: $(_unwrap_val(level.numchildren))
        |    forces: $(_unwrap_val.(level.forces))
        |    forcefile: $(level.forcefile)
        |  )
        """
    return print(io, str)
end

function Base.show(io::IO, level::HMCLevel)
    str = """
        |  HMCLevel(
        |    integrator: $(level.integrator)
        |    numsteps: $(level.numsteps)
        |    Δτ: $(level.Δτ)
        |    numchildren: $(_unwrap_val(level.numchildren))
        |    forces: $(_unwrap_val.(level.forces))
        |    forcefile: $(level.forcefile)
        |  )
        """
    print(io, str)
    return nothing
end

function level_parameters_from_dict(value::Vector)
    value_out = Vector{HMCLevelParameters}(undef, length(value))

    # Dictionary to track which forces are assigned to which levels
    force_dict = Dict{Int,Int}()

    for i in eachindex(value)
        level_params = initialize_level_parameters()
        level_dict = struct2dict(level_params)

        for (key_ii, value_ii) in value[i]
            if haskey(level_dict, key_ii)
                if !isnothing(value_ii)
                    keytype = typeof(getfield(level_params, Symbol(key_ii)))
                    setfield!(level_params, Symbol(key_ii), keytype(value_ii))
                end
            end
        end

        # Check for forces that are already assigned to other levels
        for force in level_params.forces
            if haskey(force_dict, force)
                error(
                    "Force $(force) is assigned to both level $(force_dict[force]) and ",
                    "level $(i)",
                )
            else
                force_dict[force] = i
            end
        end

        value_out[i] = deepcopy(level_params)
    end

    return value_out
end

function initialize_level_parameters()
    return HMCLevelParameters()
end

@kwdef mutable struct HMCLevelParameters
    forces::Vector{Int64} = [1]
    integrator::String = "Leapfrog"
    numsteps::Int64 = 10
end

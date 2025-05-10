struct HMCLevel{NC,TI,TFP}
    integrator::TI
    numsteps::Int64
    Δτ::Float64
    numchildren::NC
    forces::Vector{Int64} # which forces contribute to this level?
    forcefile::TFP
    function HMCLevel(
        integrator::AbstractIntegrator,
        numsteps,
        Δτ,
        forces;
        numchildren::NC=Val(0),
        hmc_logging=true,
        logdir="",
        instance=mpi_myrank(),
        distributed=false,
    )
        comm_instance = mpi_comm_instance()

        if hmc_logging && (logdir != "") && (!distributed || mpi_amroot(comm_instance))
            for ii in instance
                ext = "$(lpad(ii, 3, "0")).txt"
                _forcefile = joinpath(logdir, "hmc_force_logs_$(ext)")
                force_fp = fopen(_forcefile, "w")

                if 0 ∈ forces
                    printf(force_fp, "%-25s", "avg||F_V||")
                    printf(force_fp, "%-25s", "sup||F_V||")
                end

                if 1 ∈ forces
                    printf(force_fp, "%-25s", "avg||F_Sg||")
                    printf(force_fp, "%-25s", "sup||F_Sg||")
                end

                for i in forces
                    i ∈ (0, 1) && continue
                    printf(force_fp, "%-25s", "avg||F_Sf$(i-1)||")
                    printf(force_fp, "%-25s", "sup||F_Sf$(i-1)||")
                end

                newline(force_fp)
                fclose(force_fp)
            end

            ext = "$(lpad(instance[1], 3, "0")).txt"
            _forcefile = joinpath(logdir, "hmc_force_logs_$(ext)")
            forcefile = StaticString(_forcefile)
        end

        return new{NC,TI,TFP}(numchildren, integrator, numsteps, Δτ, forcefile)
    end
end

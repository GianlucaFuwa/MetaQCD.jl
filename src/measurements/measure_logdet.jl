struct LogDetMeasurement{T,TD,TF} <: AbstractMeasurement
    fermion_action::TD
    temp::TF # We need 1 temp fermion field per action
    LD_dict::Dict{String,Float64} # dirac_type => logdet
    filename::T
    function LogDetMeasurement(
        U::Gaugefield;
        filename="",
        dirac_type=["staggered"],
        eo_precon=false,
        flow=NoSmearing(),
        Nf=2,
        mass=0.1,
        csw=0,
        cg_tol=1e-16,
        cg_maxiters=1000,
        rhmc_order=15,
        rhmc_prec=64,
        rhmc_spectral_bound=(mass[1]^2, 6.0),
        bc_str="antiperiodic",
    )
        num_ops = length(dirac_type)
        LD_dict = Dict{String,Float64}()
        eo_precon = to_vec(eo_precon, num_ops)
        Nf = to_vec(Nf, num_ops)
        mass = to_vec(mass, num_ops)
        rhmc_order = to_vec(rhmc_order, num_ops)
        rhmc_prec = to_vec(rhmc_prec, num_ops)
        rhmc_spectral_bound = to_vec(rhmc_spectral_bound, num_ops)

        fermion_action = ntuple(length(dirac_type)) do i
            cg_filepath = if mpi_amroot(MPI_COMM_INSTANCE[]) && (filename != "")
                measdir = joinpath(splitpath(filename)[1:end-1])
                _ext = "$(lpad(MPI_INSTANCE[], 3, "0")).txt"
                joinpath(measdir, "logdet_$(i)_cg_data_$(_ext)")
            else
                ""
            end

            @level1("|    Dirac Operator: $(dirac_type[i])")
            @level1("|    Nf: $(Nf[i])")
            @level1("|    Mass: $(mass[i])")
            dirac_type == "wilson" && @level1("|    CSW: $(csw)")
            @level1("|    Even-Odd Preconditioned: $(string(eo_precon[i]))")
            @level1("|    CG Tolerance: $(cg_tol)")
            @level1("|    CG Max Iterations: $(cg_maxiters)")
            @level1("|    CG Datafile: $(cg_filepath)")
            @level1("|    RHMC Order: $(rhmc_order[i])")
            @level1("|    RHMC Precisioin: $(rhmc_prec[i])")
            @level1("|    RHMC Spectral Bound: $(string(rhmc_spectral_bound[i]))")
            LD_dict[dirac_type[i]] = 0.0
            FermionAction(
                dirac_type[i], U, mass[i];
                bc_str=bc_str,
                Nf=Nf[i],
                rhmc_spectral_bound=(rhmc_spectral_bound[i]),
                rhmc_order_md=rhmc_order[i],
                rhmc_prec_md=rhmc_prec[i],
                rhmc_order_action=rhmc_order[i],
                rhmc_prec_action=rhmc_prec[i],
                cg_tol_action=cg_tol,
                cg_tol_md=cg_tol,
                cg_maxiters_action=cg_maxiters,
                cg_maxiters_md=cg_maxiters,
                cg_filepath=cg_filepath,
                csw=csw,
            ) 
        end

        temp = ntuple(length(dirac_type)) do i
            similar(fermion_action[i].D.temp)
        end

        if !isnothing(filename) && filename != "" && mpi_amroot(mpi_comm_instance())
            if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
                fp = fopen(filename, "w")
                printf(fp, ITRJ_STR_FMT, "itrj")

                if flow == true || flow != NoSmearing()
                    printf(fp, IFLOW_STR_FMT, "iflow")
                    printf(fp, TFLOW_STR_FMT, "tflow")
                end

                for method in keys(LD_dict)
                    printf(fp, METHOD_STR_FMT, "logdet_$(method)")
                end

                newline(fp)
                fclose(fp)
            end
        else
            rpath = nothing
        end

        T = typeof(rpath)
        TD = typeof(fermion_action)
        TF = typeof(temp)
        return new{T,TD,TF}(fermion_action, temp, LD_dict, rpath)
    end
end

function LogDetMeasurement(
    U, params::LogDetParameters, filename, flow=false
)
    return LogDetMeasurement(
        U;
        filename=filename,
        flow=flow,
        dirac_type=params.type,
        Nf=params.Nf,
        mass=params.mass,
        csw=params.csw,
        eo_precon=params.eo_precon,
        cg_tol=params.cg_tol,
        cg_maxiters=params.cg_maxiters,
        rhmc_order=params.rhmc_order,
        rhmc_prec=params.rhmc_prec,
        bc_str=params.boundary_condition,
    )
end

function measure(
    m::LogDetMeasurement{T}, 
    U,
    itrj=0,
    flow=nothing;
    mpi_multi_sim=false,
    fstr="",
) where {T}
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow
    LD_dict = m.LD_dict
    ferm = m.fermion_action
    temp = m.temp

    for (i, method) in enumerate(keys(LD_dict))
        sample_pseudofermions!(temp[i], ferm[i], U)
        LD_dict[method] = calc_fermion_action(ferm[i], U, temp[i])
    end

    for method in keys(LD_dict)
        Sf = LD_dict[method]

        if !isnothing(flow)
            @level1("$itrj\t$Sf # logdet_$(method)$(fstr)_$(τ)")
        else
            @level1("$itrj\t$Sf # logdet_$(method)")
        end
    end

    if T !== Nothing
        filename = if mpi_multi_sim
            set_ext!(m.filename)
        else
            m.filename
        end

        fp = fopen(filename, "a")
        printf(fp, ITRJ_FMT, itrj)

        if !isnothing(flow)
            printf(fp, IFLOW_FMT, iflow)
            printf(fp, TFLOW_FMT, τ)
        end

        for method in keys(LD_dict)
            printf(fp, METHOD_FMT, LD_dict[method])
        end

        printf(fp, "\n")
        fclose(fp)
    end

    return LD_dict
end

struct PionCorrelatorMeasurement{T,TD,TF,CT,T1} <: AbstractMeasurement
    dirac_operator::TD
    temp::TF # We need 1 temp fermion field for propagators
    cg_temps::CT # We need 4 temp fermions for cg / 7 for bicg(stab)
    pion_corr::Vector{Float64} # One value per time slice
    cg_tol::Float64
    cg_maxiters::Int64
    cg_datafile::T1
    # mass_precon::Bool
    filename::T
    function PionCorrelatorMeasurement(
        U::Gaugefield;
        filename="",
        dirac_type="wilson",
        eo_precon=false,
        flow=NoSmearing(),
        mass=0.1,
        csw=0,
        r=1,
        cg_tol=1e-8,
        cg_maxiters=1000,
        bc_str="antiperiodic",
    )
        @level1("|    Dirac Operator: $(dirac_type)")
        @level1("|    Mass: $(mass)")
        dirac_type == "wilson" && @level1("|    CSW: $(csw)")
        @level1("|    Even-odd Preconditioned: $(string(eo_precon))")
        @level1("|    CG Tolerance: $(cg_tol)")
        @level1("|    CG Max Iterations: $(cg_maxiters)")
        @level1("|    Boundary Condition: $(bc_str)")
        NT = get_local_dims(U)[end]
        pion_corr = zeros(Float64, NT)

        if dirac_type == "staggered"
            if eo_precon
                dirac_operator = StaggeredEOPreDiracOperator(
                    U, mass; bc_str=bc_str
                )
                temp = Spinorfield(U; staggered=true)
                cg_temps = ntuple(_ -> even_odd(similar(temp)), 6)
            else
                dirac_operator = StaggeredDiracOperator(
                    U, mass; bc_str=bc_str
                )
                temp = Spinorfield(U; staggered=true)
                cg_temps = ntuple(_ -> similar(temp), 6)
            end
        elseif dirac_type == "wilson"
            dirac_operator = WilsonDiracOperator(
                U, mass; bc_str=bc_str, r=r, csw=csw
            )
            temp = Spinorfield(U)
            cg_temps = ntuple(_ -> similar(temp), 6)
        elseif dirac_type == "staggered_h1234"
            dirac_operator = StaggeredHoelblingDiracOperator{1234}(
                U, mass; bc_str=bc_str
            )
            temp = Spinorfield(U; staggered=true)
            cg_temps = ntuple(_ -> similar(temp), 6)
        elseif dirac_type == "staggered_h1342"
            dirac_operator = StaggeredHoelblingDiracOperator{1342}(
                U, mass; bc_str=bc_str
            )
            temp = Spinorfield(U; staggered=true)
            cg_temps = ntuple(_ -> similar(temp), 6)
        else
            throw(ArgumentError("Dirac operator \"$dirac_type\" is not supported"))
        end

        if !isnothing(filename) && filename != "" && mpi_amroot(mpi_comm_instance())
            rpath = SStaticString(filename)

            if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
                fp = fopen(filename, "w")
                printf(fp, ITRJ_STR_FMT, "itrj")

                if flow == true || flow != NoSmearing()
                    printf(fp, IFLOW_STR_FMT, "iflow")
                    printf(fp, TFLOW_STR_FMT, "tflow")
                end

                for it in 1:NT
                    printf(fp, METHOD_STR_FMT, "pion_corr_$(it)")
                end

                newline(fp)
                fclose(fp)
            end

            cg_filepath = if mpi_amroot(MPI_COMM_INSTANCE[]) && (filename != "")
                measdir = joinpath(splitpath(filename)[1:end-1])
                _ext = "$(lpad(MPI_INSTANCE[], 3, "0")).txt"
                joinpath(measdir, "pion_corr_cg_data_$(_ext)")
            else
                ""
            end

            cg_dataf = StaticString(cg_filepath)

            if cg_filepath != ""
                fp = fopen(cg_dataf, "w")
                printf(fp, ITRJ_STR_FMT, "iters")
                printf(fp, METHOD_STR_FMT, "res")
                newline(fp)
                fclose(fp)
            end
        else
            rpath = nothing
            cg_dataf = nothing
        end

        T = typeof(rpath)
        T1 = typeof(cg_dataf)
        TD = typeof(dirac_operator)
        TF = typeof(temp)
        CT = typeof(cg_temps)
        return new{T,TD,TF,CT,T1}(
            dirac_operator, temp, cg_temps, pion_corr, cg_tol, cg_maxiters, cg_dataf, rpath
        )
    end
end

function PionCorrelatorMeasurement(
    U, params::PionCorrelatorParameters, filename, flow=false
)
    return PionCorrelatorMeasurement(
        U;
        filename=filename,
        flow=flow,
        dirac_type=params.dirac_type,
        mass=params.mass,
        csw=params.csw,
        eo_precon=params.eo_precon,
        cg_tol=params.cg_tol,
        cg_maxiters=params.cg_maxiters,
        bc_str=params.boundary_condition,
    )
end

function measure(
    m::PionCorrelatorMeasurement{T}, 
    U,
    itrj=0,
    flow=nothing;
    mpi_multi_sim=false,
    kwargs...,
) where {T}
    DU = m.dirac_operator(U)
    pion_correlators_avg!(
        m.pion_corr, DU, m.temp, m.cg_temps, m.cg_tol, m.cg_maxiters, m.cg_datafile
    )
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

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

        for value in m.pion_corr
            printf(fp, METHOD_FMT, value)
        end

        printf(fp, "\n")
        fclose(fp)
    end

    return m.pion_corr
end

"""
    pion_correlators_avg!(pion_corr, D, ψ, cg_temps, cg_tol, cg_maxiters)

Calculate the pion correlators for a given configuration and store the result for each
time slice in the vector `pion_corr`. \\
We follow the procedure outlined in DOI: 10.1007/978-3-642-01850-3 (Gattringer) pages
135-136 using a point source for each dirac and color index from the origin
"""
function pion_correlators_avg!(pion_corr, D, ψ, cg_temps, tol, maxiters, datafile)
    check_dims(D.U, ψ, cg_temps...)
    M = is_distributed(D.U)
    B = get_backend(D.U)
    NX, NY, NZ, NT = size(ψ)
    xrange, yrange, zrange, _ = D.U.topology.bulk_sites.indices
    itr = CartesianIndices((xrange, yrange, zrange))
    @assert length(pion_corr) == NT

    # Point source at origin
    source = SiteCoords(1, 1, 1, 1)
    # Get temporary arrays for cg solver
    propagator, temps... = cg_temps
    pion_corr .= 0.0

    for a in 1:3
        for μ in 1:num_dirac(ψ)
            ones!(propagator)
            set_source!(ψ, source, a, μ)
            solve_dirac!(
                propagator, D, ψ, temps...; tol, maxiters, datafile
            )

            for it in 1:NT
                cit = 0.0

                cit = parallelfor_sum(itr, 0.0, B, Val(M), (), (), (propagator,)) do ci, xyz, (propagator,)
                    ix, iy, iz = xyz.I
                    ci += real(
                        cdot(propagator[ix, iy, iz, it], propagator[ix, iy, iz, it])
                    )
                end

                pion_corr[it] += distributed_reduce(cit, +, D.U)
            end
        end
    end

    Λₛ = NX * NY * NZ

    for it in 1:NT
        pion_corr[it] /= Λₛ
    end

    return nothing
end

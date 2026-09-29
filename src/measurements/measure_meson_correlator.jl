abstract type AbstractMesonCorrelator end
struct PseudoPseudo <: AbstractMesonCorrelator end
struct AxialPseudo <: AbstractMesonCorrelator end

function meson_correlator_from_string(type)
    return if lowercase(type) == "pseudo_pseudo"
        PseudoPseudo()
    elseif lowercase(type) == "axial_pseudo"
        AxialPseudo()
    else
        error("Correlator type $(type) not supported")
    end
end

struct MesonCorrelatorMeasurement{T,TC,TD,TF,CT,D,T1} <: AbstractMeasurement
    type::TC
    dirac_operator::TD
    temp::TF # We need 1 temp fermion field for propagators
    cg_temps::CT # We need 4 temp fermions for cg / 7 for bicg(stab)
    corr::Dict{Tuple{Any,Int64},Vector{Float64}} # One value per time slice
    directions::D
    cg_tol::Float64
    cg_maxiters::Int64
    cg_datafile::T1
    # mass_precon::Bool
    filename::T
    function MesonCorrelatorMeasurement(
        U::Gaugefield;
        filename="",
        type=["pseudo_pseudo"],
        dirac_type="wilson",
        directions=[4],
        eo_precon=false,
        flow=NoSmearing(),
        mass=0.1,
        csw=0,
        r=1,
        cg_tol=1e-8,
        cg_maxiters=1000,
        bc_str="antiperiodic",
    )
        @level1("|    Type: $(string(type))")
        @level1("|    Dirac Operator: $(dirac_type)")
        @level1("|    Mass: $(mass)")
        @level1("|    Directions: $(string(directions))")
        dirac_type == "wilson" && @level1("|    CSW: $(csw)")
        @level1("|    Even-odd Preconditioned: $(string(eo_precon))")
        @level1("|    CG Tolerance: $(cg_tol)")
        @level1("|    CG Max Iterations: $(cg_maxiters)")
        @level1("|    Boundary Condition: $(bc_str)")

        corrtype = ntuple(i -> meson_correlator_from_string(type[i]), length(type))
        NT = size(U)[end]
        corr = Dict{Tuple{Any,Int64},Vector{Float64}}()
        for t in corrtype
            for dir in directions
                corr[(t, dir)] = zeros(Float64, size(U)[dir])
            end
        end

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
            istr = lpad(MPI_INSTANCE[], 3, "0")
            dir = joinpath("/", split(filename, "/")[1:end-1]...)
            rpath = Dict{Tuple{Any,Int64},SStaticString}()
            # cg_dataf = Dict{Any,SStaticString}()

            for (itype, t) in enumerate(corrtype)
                for μ in directions
                    rpath[(t, μ)] = SStaticString(joinpath(dir, "meson_correlator_$(type[itype])_$(μ)_$(istr).txt"))

                    if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
                        filename = joinpath(dir, "meson_correlator_$(type[itype])_$(μ)_$(istr).txt")
                        fp = fopen(filename, "w")
                        printf(fp, ITRJ_STR_FMT, "itrj")

                        if flow == true || flow !== NoSmearing()
                            printf(fp, IFLOW_STR_FMT, "iflow")
                            printf(fp, TFLOW_STR_FMT, "tflow")
                        end

                        for it in 1:NT
                            printf(fp, METHOD_STR_FMT, "$(type[itype])_corr_$(it)")
                        end

                        newline(fp)
                        fclose(fp)
                    end
                end
            end

            cg_filepath = if mpi_amroot(MPI_COMM_INSTANCE[]) && (filename != "")
                measdir = joinpath(splitpath(filename)[1:end-1])
                _ext = "$(lpad(MPI_INSTANCE[], 3, "0")).txt"
                joinpath(measdir, "meson_correlator_cg_data_$(_ext)")
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

        directions = ntuple(i -> Val(directions[i]), length(directions))
        T = typeof(rpath)
        TC = typeof(corrtype)
        T1 = typeof(cg_dataf)
        TD = typeof(dirac_operator)
        TF = typeof(temp)
        CT = typeof(cg_temps)
        D = typeof(directions)
        return new{T,TC,TD,TF,CT,D,T1}(
            corrtype,
            dirac_operator,
            temp,
            cg_temps,
            corr,
            directions,
            cg_tol,
            cg_maxiters,
            cg_dataf,
            rpath,
        )
    end
end

function MesonCorrelatorMeasurement(
    U, params::MesonCorrelatorParameters, filename, flow=false
)
    return MesonCorrelatorMeasurement(
        U;
        filename=filename,
        type=params.type,
        flow=flow,
        dirac_type=params.dirac_type,
        mass=params.mass,
        directions=params.directions,
        csw=params.csw,
        eo_precon=params.eo_precon,
        cg_tol=params.cg_tol,
        cg_maxiters=params.cg_maxiters,
        bc_str=params.boundary_condition,
    )
end

function measure(
    m::MesonCorrelatorMeasurement{T,TC}, 
    U,
    itrj=0,
    flow=nothing;
    mpi_multi_sim=false,
    kwargs...,
) where {T,TC}
    DU = m.dirac_operator(U)
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    correlator_avg!(
        m.type,
        m.corr,
        DU,
        m.temp,
        m.directions,
        m.cg_temps,
        m.cg_tol,
        m.cg_maxiters,
        m.cg_datafile,
    )

    if T !== Nothing
        for t in m.type
            for μ in _unwrap_val.(m.directions)
                filename = if mpi_multi_sim
                    set_ext!(m.filename[(t, μ)])
                else
                    m.filename[(t, μ)]
                end

                fp = fopen(filename, "a")
                printf(fp, ITRJ_FMT, itrj)

                if !isnothing(flow)
                    printf(fp, IFLOW_FMT, iflow)
                    printf(fp, TFLOW_FMT, τ)
                end

                for value in m.corr[(t, μ)]
                    printf(fp, METHOD_FMT, value)
                end

                printf(fp, "\n")
                fclose(fp)
            end
        end
    end

    return m.corr
end

"""
    correlator_avg!(type, corr, D, ψ, cg_temps, cg_tol, cg_maxiters)

Calculate the `type` correlators for a given configuration and store the result for each
time slice in the vector `corr`. \\
We follow the procedure outlined in DOI: 10.1007/978-3-642-01850-3 (Gattringer) pages
135-136 using a point source for each dirac and color index from the origin
"""
function correlator_avg!(
    type, corr, D, ψ, directions, cg_temps, tol, maxiters, datafile
)
    check_dims(D.U, ψ, cg_temps...)
    global_dims = size(ψ)
    local_ranges = D.U.topology.bulk_sites.indices

    for t in type
        for dir in _unwrap_val.(directions)
            @assert length(corr[(t, dir)]) == global_dims[dir]
            corr[(t, dir)] .= 0.0
        end
    end

    # Point source at origin
    source = SiteCoords(1, 1, 1, 1)
    # Get temporary arrays for cg solver
    propagator, temps... = cg_temps

    for dir in directions
        itr = CartesianIndices((local_ranges[[(directions .!= _unwrap_val(dir))...]]))

        for a in 1:3
            for μ in 1:num_dirac(ψ)
                ones!(propagator)
                set_source!(ψ, source, a, μ)
                solve_dirac!(
                    propagator, D, ψ, temps...; tol, maxiters, datafile
                )

                for t in type
                    if t == PseudoPseudo()
                        calc_pseudo_pseudo_corr(
                            D, corr, propagator, dir, global_dims[_unwrap_val(dir)], itr
                        )
                    elseif t == AxialPseudo()
                        calc_axial_pseudo_corr(
                            D, corr, propagator, dir, global_dims[_unwrap_val(dir)], itr
                        )
                    end
                end
            end
        end
    end

    for t in type
        for dir in _unwrap_val.(directions)
            Λₛ = prod(global_dims[[(directions .!= dir)...]])
            corr[(t, dir)] ./= Λₛ
        end
    end

    return nothing
end

# function correlator_avg!(
#     ::AxialPseudo, corr, D, ψ, directions, cg_temps, tol, maxiters, datafile
# )
#     check_dims(D.U, ψ, cg_temps...)
#     global_dims = size(ψ)
#     local_ranges = D.U.topology.bulk_sites.indices
#
#     for dir in _unwrap_val.(directions)
#         @assert length(corr[(AxialPseudo(), dir)]) == global_dims[dir]
#         corr[(AxialPseudo(), dir)] .= 0.0
#     end
#
#     # Point source at origin
#     source = SiteCoords(1, 1, 1, 1)
#     # Get temporary arrays for cg solver
#     propagator, temps... = cg_temps
#
#     for dir in directions
#         itr = CartesianIndices((local_ranges[[(directions .!= _unwrap_val(dir))...]]))
#
#         for a in 1:3
#             for μ in 1:num_dirac(ψ)
#                 ones!(propagator)
#                 set_source!(ψ, source, a, μ)
#                 solve_dirac!(
#                     propagator, D, ψ, temps...; tol, maxiters, datafile
#                 )
#
#                 calc_axial_pseudo_corr(
#                     D, corr, propagator, dir, global_dims[_unwrap_val(dir)], itr
#                 )
#             end
#         end
#     end
#
#     for dir in _unwrap_val.(directions)
#         Λₛ = prod(global_dims[[(directions .!= dir)...]])
#         corr[(AxialPseudo(), dir)] ./= Λₛ
#     end
#
#     return nothing
# end

function calc_pseudo_pseudo_corr(D, corr, propagator, ::Val{μ}, L, itr) where {μ}
    M = is_distributed(D.U)
    B = get_backend(D.U)

    for ii in 1:L
        cit = 0.0

        cit = parallelfor_sum(itr, 0.0, B, Val(M), (), (), (propagator,)) do ci, xyz, (propagator,)
            coords = xyz.I
            site = SiteCoords(ntuple(d -> d<μ ? coords[d] : (d==μ ? ii : coords[d-1]), 4))
            ci += real(dot(propagator[site], propagator[site]))
        end

        corr[(PseudoPseudo(), μ)][ii] += distributed_reduce(cit, +, D.U)
    end

    return nothing
end

function calc_axial_pseudo_corr(D, corr, propagator, ::Val{μ}, L, itr) where {μ}
    U = D.U
    M = is_distributed(U)
    B = get_backend(U)

    for ii in 1:L
        cit = 0.0

        cit = parallelfor_sum(itr, 0.0, B, Val(M), (), (), (propagator,)) do ci, xyz, (propagator,)
            coords = xyz.I
            site = SiteCoords(ntuple(d -> d<μ ? coords[d] : (d==μ ? ii : coords[d-1]), 4))
            Nμ = axes(propagator, μ)
            siteμ⁺ = move(site, μ, 1, Nμ)
            η = staggered_η(Val(μ), site, Float32)
            ϵ = staggered_ϵ(site, Float32)
            ci +=  ϵ * η * real(dot(propagator[site], cmvmul(U[μ, siteμ⁺], propagator[site])))
        end

        corr[(AxialPseudo(), μ)][ii] += distributed_reduce(cit, +, D.U)
    end

    return nothing
end

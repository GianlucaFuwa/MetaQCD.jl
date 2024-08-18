struct PionCorrelatorMeasurement{T,TD,TF,CT} <: AbstractMeasurement
    dirac_operator::TD
    temp::TF # We need 1 temp fermion field for propagators
    cg_temps::CT # We need 4 temp fermions for cg / 7 for bicg(stab)
    pion_corr::Vector{Float64} # One value per time slice
    cg_tol::Float64
    cg_maxiters::Int64
    # mass_precon::Bool
    filename::T
    function PionCorrelatorMeasurement(
        U::Gaugefield;
        filename="",
        dirac_type="wilson",
        eo_precon=false,
        flow=false,
        mass=0.1,
        csw=0,
        r=1,
        cg_tol=1e-16,
        cg_maxiters=1000,
        anti_periodic=true,
    )
        @level1("|    Dirac Operator: $(dirac_type)")
        @level1("|    Mass: $(mass)")
        dirac_type == "wilson" && @level1("|    CSW: $(csw)")
        @level1("|    Even-odd preconditioned: $(eo_precon)")
        @level1("|    CG Tolerance: $(cg_tol)")
        @level1("|    CG Max Iterations: $(cg_maxiters)")
        NT = dims(U)[end]
        pion_corr = zeros(Float64, NT)

        if dirac_type == "staggered"
            if eo_precon
                dirac_operator = StaggeredEOPreDiracOperator(
                    U, mass; anti_periodic=anti_periodic
                )
                temp = Spinorfield(U; staggered=true)
                cg_temps = ntuple(_ -> even_odd(similar(temp)), 6)
            else
                dirac_operator = StaggeredDiracOperator(
                    U, mass; anti_periodic=anti_periodic
                )
                temp = Spinorfield(U; staggered=true)
                cg_temps = ntuple(_ -> similar(temp), 6)
            end
        elseif dirac_type == "wilson"
            dirac_operator = WilsonDiracOperator(
                U, mass; anti_periodic=anti_periodic, r=r, csw=csw
            )
            temp = Spinorfield(U)
            cg_temps = ntuple(_ -> Spinorfield(temp), 6)
        else
            throw(ArgumentError("Dirac operator \"$dirac_type\" is not supported"))
        end

        if filename !== nothing && filename != ""
            path = filename * MYEXT
            rpath = StaticString(path)
            header = ""

            if flow
                header *= @sprintf("%-11s%-7s%-9s", "itrj", "iflow", "tflow")
            else
                header *= @sprintf("%-11s", "itrj")
            end

            for it in 1:NT
                header *= @sprintf("%-25s", "pion_corr_$(it)")
            end

            open(path, "w") do fp
                println(fp, header)
            end
        else
            rpath = nothing
        end

        T = typeof(rpath)
        TD = typeof(dirac_operator)
        TF = typeof(temp)
        CT = typeof(cg_temps)
        return new{T,TD,TF,CT}(
            dirac_operator, temp, cg_temps, pion_corr, cg_tol, cg_maxiters, rpath
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
        anti_periodic=params.anti_periodic,
    )
end

function measure(
    m::PionCorrelatorMeasurement{T}, U, myinstance, itrj, flow=nothing
) where {T}
    pion_correlators_avg!(
        m.pion_corr, m.dirac_operator(U), m.temp, m.cg_temps, m.cg_tol, m.cg_maxiters
    )
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    if T !== Nothing
        filename = set_ext!(m.filename, myinstance)
        fp = fopen(filename, "a")
        printf(fp, "%-11i", itrj)

        if !isnothing(flow)
            printf(fp, "%-7i", iflow)
            printf(fp, "%-9.5f", τ)
        end

        for value in m.pion_corr
            printf(fp, "%+-25.15E", value)
        end

        printf(fp, "\n")
        fclose(fp)
    end

    return m.pion_corr
end

"""
    pion_correlators_avg!(pion_corr, D, ψ, cg_temps, cg_tol, cg_maxiters)

Calculate the pion correlators for a given configuration and store the result for each
time slice in `pion_corr`. \\
We follow the procedure outlined in DOI: 10.1007/978-3-642-01850-3 (Gattringer) pages
135-136 using point sources for each dirac and color index all starting from the origin
"""
function pion_correlators_avg!(pion_corr, D, ψ, cg_temps, cg_tol, cg_maxiters)
    check_dims(D.U, ψ, cg_temps...)
    NX, NY, NZ, NT = dims(ψ)
    @assert length(pion_corr) == NT
    source = SiteCoords(1, 1, 1, 1)
    propagator, temps... = cg_temps
    pion_corr .= 0.0

    for a in 1:ψ.NC
        for μ in 1:ψ.ND
            ones!(propagator)
            set_source!(ψ, source, a, μ)
            solve_dirac!(propagator, D, ψ, temps...; tol=cg_tol, maxiters=cg_maxiters)
            for it in 1:NT
                cit = 0.0
                @batch reduction = (+, cit) for iz in 1:NZ
                    for iy in 1:NY
                        for ix in 1:NX
                            cit += real(
                                cdot(propagator[ix, iy, iz, it], propagator[ix, iy, iz, it])
                            )
                        end
                    end
                end
                pion_corr[it] += cit
            end
        end
    end

    Λₛ = NX * NY * NZ
    for it in 1:NT
        pion_corr[it] /= Λₛ
    end

    return nothing
end

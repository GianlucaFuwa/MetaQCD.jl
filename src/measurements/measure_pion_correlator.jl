struct PionCorrelatorMeasurement{T,TD,TF,CT,RT,RI} <: AbstractMeasurement
    dirac_operator::TD
    temp::TF # We need 1 temp fermion field for propagators
    cg_temps::CT # We need 4 temp fermions for cg / 7 for bicg(stab)
    rhmc_temps1::RT # this holds the results of multishift cg
    rhmc_temps2::RT # this holds the basis vectors in multishift cg
    pion_dict::Dict{Int64,Float64} # One value per time slice
    rhmc_info::RI
    cg_tol::Float64
    cg_maxiters::Int64
    # mass_precon::Bool
    fp::T # file pointer
    function PionCorrelatorMeasurement(
        U::Gaugefield;
        filename="",
        printvalues=false,
        dirac_type="wilson",
        eo_precon=false,
        flow=false,
        mass=0.1,
        Nf=1,
        csw=0,
        r=1,
        cg_tol=1e-16,
        cg_maxiters=1000,
        rhmc_order=15,
        rhmc_prec=42,
        anti_periodic=true,
    )
        pion_dict = Dict{Int64,Float64}()
        NT = dims(U)[end]

        for it in 1:NT
            pion_dict[it] = 0.0
        end

        if dirac_type == "staggered"
            if eo_precon
                error(
                    "Even-odd preconditioned staggered not yet supported in pion correlator"
                )
                dirac_operator = StaggeredEOPreDiracOperator(
                    U, mass; anti_periodic=anti_periodic
                )
                temp = Fermionfield(U; staggered=true)
                cg_temps = ntuple(_ -> even_odd(similar(temp)), 6)
                # TODO
                power = Nf//4
                # rhmc_info = RHMCParams(power; n=rhmc_order, precision=rhmc_prec)
                rhmc_info = rhmc_temps1 = rhmc_temps2 = nothing
            else
                dirac_operator = StaggeredDiracOperator(
                    U, mass; anti_periodic=anti_periodic
                )
                temp = Fermionfield(U; staggered=true)
                cg_temps = ntuple(_ -> similar(temp), 6)
                power = Nf//4
                # rhmc_info = RHMCParams(power; n=rhmc_order, precision=rhmc_prec)
                rhmc_info = rhmc_temps1 = rhmc_temps2 = nothing
            end
        elseif dirac_type == "wilson"
            dirac_operator = WilsonDiracOperator(
                U, mass; anti_periodic=anti_periodic, r=r, csw=csw
            )
            temp = Fermionfield(U)
            cg_temps = ntuple(_ -> Fermionfield(temp), 6)
            rhmc_info = rhmc_temps1 = rhmc_temps2 = nothing
        else
            throw(ArgumentError("Dirac operator \"$dirac_type\" is not supported"))
        end

        if printvalues
            fp = open(filename, "w")
            header = ""

            if flow
                header *= @sprintf("%-9s\t%-7s\t%-9s", "itrj", "iflow", "tflow")
            else
                header *= @sprintf("%-9s", "itrj")
            end

            for it in 1:NT
                header *= @sprintf("\t%-22s", "pion_corr_$(it)")
            end

            println(fp, header)
        else
            fp = nothing
        end

        T = typeof(fp)
        TD = typeof(dirac_operator)
        TF = typeof(temp)
        CT = typeof(cg_temps)
        RT = typeof(rhmc_temps1)
        RI = typeof(rhmc_info)
        return new{T,TD,TF,CT,RT,RI}(
            dirac_operator,
            temp,
            cg_temps,
            rhmc_temps1,
            rhmc_temps2,
            pion_dict,
            rhmc_info,
            cg_tol,
            cg_maxiters,
            fp,
        )
    end
end

function PionCorrelatorMeasurement(
    U, params::PionCorrelatorParameters, filename, flow=false
)
    return PionCorrelatorMeasurement(
        U;
        filename=filename,
        printvalues=true,
        flow=flow,
        dirac_type=params.dirac_type,
        mass=params.mass,
        # Nf=params.Nf,
        # r=params.r,
        cg_tol=params.cg_tol,
        cg_maxiters=params.cg_maxiters,
        anti_periodic=params.anti_periodic,
    )
end

function measure(m::PionCorrelatorMeasurement{T}, U; additional_string="") where {T}
    measurestring = ""
    printstring = @sprintf("%-9s", additional_string)

    pion_correlators_avg!(
        m.pion_dict,
        m.dirac_operator(U),
        m.temp,
        m.cg_temps,
        m.cg_tol,
        m.cg_maxiters,
    )

    if T ≡ IOStream
        for value in values(m.pion_dict)
            svalue = @sprintf("%+-22.15E", value)
            printstring *= "\t$svalue"
        end

        measurestring = printstring
        println(m.fp, measurestring)
        flush(m.fp)
        measurestring *= " # pion_correlator"
    end

    output = MeasurementOutput(m.pion_dict, measurestring)
    return output
end

"""
    pion_correlators_avg!(dict, D, ψ, cg_temps, cg_tol, cg_maxiters)

Calculate the pion correlators for a given configuration and store the result for each
time slice in `dict`. \\
We follow the procedure outlined in DOI: 10.1007/978-3-642-01850-3 (Gattringer) pages
135-136 using point sources for each dirac and color index all starting from the origin
"""
function pion_correlators_avg!(dict, D, ψ, cg_temps, cg_tol, cg_maxiters)
    check_dims(D.U, ψ, cg_temps...)
    NX, NY, NZ, NT = dims(ψ)
    @assert length(dict) == NT
    source = SiteCoords(1, 1, 1, 1)
    propagator, temps... = cg_temps

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
                dict[it] = cit
            end
        end
    end

    Λₛ = NX * NY * NZ
    for it in 1:NT
        dict[it] /= Λₛ
    end

    return nothing
end

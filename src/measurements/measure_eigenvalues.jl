struct EigenvaluesMeasurement{T,TA,TD} <: AbstractMeasurement
    arnoldi::TA
    dirac_operator::TD
    vals::Vector{ComplexF64}
    nev::Int64
    tol::Float64
    mindim::Int64
    maxdim::Int64
    restarts::Int64
    which::Symbol
    ddaggerd::Bool
    fp::T # file pointer
    function EigenvaluesMeasurement(
        U::Gaugefield;
        filename="",
        printvalues=false,
        dirac_type="wilson",
        eo_precon=false,
        flow=false,
        mass=0.1,
        csw=0,
        r=1,
        anti_periodic=true,
        nev = 10,
        which = "LM",
        tol = sqrt(eps(real(Float64))),
        mindim = max(10, nev),
        maxdim = max(20, 2nev),
        restarts = 200,
        ddaggerd = false,
    )
        if dirac_type == "staggered"
            if eo_precon
                dirac_operator = StaggeredEOPreDiracOperator(
                    U, mass; anti_periodic=anti_periodic
                )
            else
                dirac_operator = StaggeredDiracOperator(
                    U, mass; anti_periodic=anti_periodic
                )
            end
        elseif dirac_type == "wilson"
            dirac_operator = WilsonDiracOperator(
                U, mass; anti_periodic=anti_periodic, r=r, csw=csw
            )
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

            if which == "LM" || which == "LR" || which == "LI"
                vals = zeros(ComplexF64, nev)
            elseif which == "SM" || which == "SR" || which == "SI"
                vals = zeros(ComplexF64, nev)
            elseif which == "LSM"
                vals = zeros(ComplexF64, 2nev)
            else
                error("\"which\" in eigenvalue measurement can only be LM, LR, LI, SM, SR, SI or LSM. Was $which")
            end
            
            for i in eachindex(vals)
                header *= @sprintf("\t%-22s\t%-22s", "eig_re_$(i)", "eig_im_$(i)")
            end

            println(fp, header)
        else
            fp = nothing
        end

        arnoldi = ArnoldiWorkspaceMeta(dirac_operator, maxdim)

        T = typeof(fp)
        TA = typeof(arnoldi)
        TD = typeof(dirac_operator)
        return new{T,TA,TD}(
            arnoldi,
            dirac_operator,
            vals,
            nev,
            tol,
            mindim,
            maxdim,
            restarts,
            Symbol(which),
            ddaggerd,
            fp,
        )
    end
end

function EigenvaluesMeasurement(U, params::EigenvaluesParameters, filename, flow=false)
    return EigenvaluesMeasurement(
        U;
        filename=filename,
        printvalues=true,
        flow=flow,
        dirac_type=params.dirac_type,
        mass=params.mass,
        csw=params.csw,
        anti_periodic=params.anti_periodic,
        eo_precon=params.eo_precon,
        nev=params.nev,
        which=params.which,
        mindim=params.mindim,
        maxdim=params.maxdim,
        tol=params.tol,
        restarts=params.restarts,
        ddaggerd=params.ddaggerd,
    )
end

function measure(m::EigenvaluesMeasurement{T}, U; additional_string="") where {T}
    measurestring = ""
    printstring = @sprintf("%-9s", additional_string)
    vals = m.vals

    if m.which == :LSM
        view(vals, 1:m.nev) .= try
            get_eigenvalues(
                U,
                m.dirac_operator,
                m.arnoldi;
                nev=m.nev,
                which=:LM, 
                tol=m.tol,
                mindim=m.mindim,
                restarts=m.restarts,
                ddaggerd=m.ddaggerd,
            )
        catch _
            @level1("@Warning: Eigenvalue calculation did not converge, will be set to 0")
            0
        end
        view(vals, m.nev+1:2m.nev) .= try
            get_eigenvalues(
                U,
                m.dirac_operator,
                m.arnoldi;
                nev=m.nev,
                which=:SM, 
                tol=m.tol,
                mindim=m.mindim,
                restarts=m.restarts,
                ddaggerd=m.ddaggerd,
            )
        catch _
            @level1("@Warning: Eigenvalue calculation did not converge, will be set to 0")
            0
        end
    else
        vals .= try 
            get_eigenvalues(
                U,
                m.dirac_operator,
                m.arnoldi;
                nev=m.nev,
                which=m.which, 
                tol=m.tol,
                mindim=m.mindim,
                restarts=m.restarts,
                ddaggerd=m.ddaggerd,
            )
        catch _
            @level1("@Warning: Eigenvalue calculation did not converge, will be set to 0")
            0
        end
    end

    if T ≡ IOStream
        for value in vals
            svalue = @sprintf("%+-22.15E\t%+-22.15E", real(value), imag(value))
            printstring *= "\t$svalue"
        end

        measurestring = printstring
        println(m.fp, measurestring)
        flush(m.fp)
    end

    output = MeasurementOutput(vals, "")
    return output
end

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
    filename::T
    function EigenvaluesMeasurement(
        U::Gaugefield;
        filename::Union{String,Nothing}=nothing,
        dirac_type="wilson",
        eo_precon=false,
        flow=false,
        mass=0.1,
        csw=0,
        r=1,
        bc_str="antiperiodic",
        nev = 10,
        which = "LM",
        tol = sqrt(eps(real(Float64))),
        mindim = max(10, nev),
        maxdim = max(20, 2nev),
        restarts = 200,
        ddaggerd = false,
    )
        @level1("|    Dirac Operator: $(dirac_type)")
        @level1("|    Mass: $(mass)")
        dirac_type == "wilson" && @level1("|    CSW: $(csw)")
        @level1("|    Even-odd preconditioned: $(eo_precon)")
        @level1("|    Number of Eigenvalues: $(nev)")
        @level1("|    Which Eigenvalues: $(which)")
        @level1("|    Min. Krylov dimension: $(mindim)")
        @level1("|    Max. Krylov dimension: $(maxdim)")
        @level1("|    Number of restarts: $(restarts)")
        @level1("|    Use D†D: $(ddaggerd)")
        @level1("|    Boundary Condition: $(bc_str)")
        if dirac_type == "staggered"
            if eo_precon
                dirac_operator = StaggeredEOPreDiracOperator(
                    U, mass; bc_str=bc_str
                )
                @level1("@Warn Eigenvalues with \"eo_precon=true\" defaults to DdaggerD")
                ddaggerd = true
            else
                dirac_operator = StaggeredDiracOperator(
                    U, mass; bc_str=bc_str
                )
            end
        elseif dirac_type == "wilson"
            if eo_precon
                error("Even-odd preconditioned Wilson Operator not supported in Eigenvalues")
            else
                dirac_operator = WilsonDiracOperator(
                    U, mass; bc_str=bc_str, r=r, csw=csw
                )
            end
        else
            throw(ArgumentError("Dirac operator \"$dirac_type\" is not supported"))
        end

        if !isnothing(filename) && filename != ""
            path = filename * MYEXT
            rpath = StaticString(path)
            header = ""

            if flow
                header *= @sprintf("%-11s%-7s%-9s", "itrj", "iflow", "tflow")
            else
                header *= @sprintf("%-11s", "itrj")
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
                header *= @sprintf("%-25s%-25s", "eig_re_$(i)", "eig_im_$(i)")
            end

            if mpi_amroot()
                open(path, "w") do fp
                    println(fp, header)
                end
            end
        else
            rpath = nothing
        end

        arnoldi = ArnoldiWorkspaceMeta(dirac_operator, maxdim)

        T = typeof(rpath)
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
            rpath,
        )
    end
end

function EigenvaluesMeasurement(U, params::EigenvaluesParameters, filename, flow=false)
    return EigenvaluesMeasurement(
        U;
        filename=filename,
        flow=flow,
        dirac_type=params.dirac_type,
        mass=params.mass,
        csw=params.csw,
        bc_str=params.boundary_condition,
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

function measure(
    m::EigenvaluesMeasurement{T}, U, myinstance, itrj, flow=nothing
) where {T}
    vals = m.vals
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

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

    if mpi_amroot()
        if T !== Nothing
            filename = set_ext!(m.filename, myinstance)
            fp = fopen(filename, "a")
            @printf(fp, "%-11i", itrj)

            if !isnothing(flow)
                @printf(fp, "%-7i", iflow)
                @printf(fp, "%-9.5f", τ)
            end

            for value in vals
                @printf(fp, "%+-25.15E", real(value))
                @printf(fp, "%+-25.15E", imag(value))
            end

            @printf(fp, "\n")
            fclose(fp)
        end
    end

    return vals
end

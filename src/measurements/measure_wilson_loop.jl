struct WilsonLoopMeasurement{T} <: AbstractMeasurement
    WL::Matrix{Float64} # (R, T) => value
    Tmax::Int64 # maximum width of the Wilson loop
    Rmax::Int64 # maximum length of the Wilson loop
    filename::T
    function WilsonLoopMeasurement(
        U::Gaugefield; filename="", Rmax=4, Tmax=4, flow=false
    )
        @assert !is_distributed(U) "Wilson loop not supported for distributed fields"
        @level1("|    Maximum Extends: $Tmax x $Rmax (only even extends are measured for now)")
        @level1("|    @info: Wilson loop measurements are not printed to console")
        WL = zeros(Rmax, Tmax)

        if !isnothing(filename) && filename != ""
            path = filename * MYEXT
            rpath = StaticString(path)
            header = ""

            if flow
                header *= @sprintf("%-11s%-7s%-9s", "itrj", "iflow", "tflow")
            else
                header *= @sprintf("%-11s", "itrj")
            end

            for iT in 1:Tmax
                for iR in 1:Rmax
                    header *= @sprintf("%-25s", "wilson_loop_$(iR)x$(iT)")
                end
            end

            if mpi_amroot()
                open(path, "w") do fp
                    println(fp, header)
                end
            end
        else
            rpath = nothing
        end

        T = typeof(rpath)
        return new{T}(WL, Tmax, Rmax, rpath)
    end
end

function WilsonLoopMeasurement(U, params::WilsonLoopParameters, filename, flow=false)
    return WilsonLoopMeasurement(
        U;
        filename=filename,
        Rmax=params.Rmax,
        Tmax=params.Tmax,
        flow=flow,
    )
end

function measure(
    m::WilsonLoopMeasurement{T}, U, myinstance, itrj, flow=nothing
) where {T}
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    for iT in 1:(m.Tmax)
        for iR in 1:(m.Rmax)
            WL = wilsonloop(U, iR, iT) / (18.0U.NV)
            m.WL[iR, iT] = WL
        end
    end

    if mpi_amroot()
        if T !== Nothing
            filename = set_ext!(m.filename, myinstance)
            fp = fopen(filename, "a")
            printf(fp, "%-11i", itrj)

            if !isnothing(flow)
                printf(fp, "%-7i\t", iflow)
                printf(fp, "%-9.5f\t", τ)
            end

            for iT in 1:(m.Tmax)
                for iR in 1:(m.Rmax)
                    printf(fp, "%+-25.15E", m.WL[iR, iT]::Float64)
                end
            end

            printf(fp, "\n")
            fclose(fp)
        end
    end

    return m.WL
end

struct WilsonLoopMeasurement{T} <: AbstractMeasurement
    WL::Matrix{Float64} # (R, T) => value
    Tmax::Int64 # maximum width of the Wilson loop
    Rmax::Int64 # maximum length of the Wilson loop
    filename::T
    function WilsonLoopMeasurement(
        U::Gaugefield; filename="", Rmax=4, Tmax=4, flow=NoSmearing()
    )
        # @assert !is_distributed(U) "Wilson loop not supported for distributed fields"
        @level1("|    Maximum Extends: $Tmax x $Rmax")
        @level1("|    @info: Wilson loop measurements are not printed to console")
        WL = zeros(Rmax, Tmax)

        if !isnothing(filename) && filename != "" && mpi_amroot(mpi_comm_instance())
            rpath = SStaticString(filename)

            if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
                fp = fopen(filename, "w")
                printf(fp, ITRJ_STR_FMT, "itrj")

                if flow == true || flow !== NoSmearing()
                    printf(fp, IFLOW_STR_FMT, "iflow")
                    printf(fp, TFLOW_STR_FMT, "tflow")
                end

                for iT in 1:Tmax
                    for iR in 1:Rmax
                        printf(fp, METHOD_STR_FMT, "wilson_loop_$(iR)x$(iT)")
                    end
                end

                newline(fp)
                fclose(fp)
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
    m::WilsonLoopMeasurement{T},
    U,
    itrj=0,
    flow=nothing;
    mpi_multi_sim=false,
    kwargs...,
) where {T}
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    for iT in 1:(m.Tmax)
        for iR in 1:(m.Rmax)
            WL = wilsonloop(U, iR, iT) / (18.0length(U))
            m.WL[iR, iT] = WL
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

        for iT in 1:(m.Tmax)
            for iR in 1:(m.Rmax)
                printf(fp, METHOD_FMT, m.WL[iR, iT]::Float64)
            end
        end

        printf(fp, "\n")
        fclose(fp)
    end

    return m.WL
end

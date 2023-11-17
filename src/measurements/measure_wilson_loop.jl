import ..Gaugefields: wilsonloop

struct WilsonLoopMeasurement{T} <: AbstractMeasurement
    WL_dict::Dict{NTuple{2,Int64}, Float64}
    Tmax::Int64
    Rmax::Int64
    fp::T

    function WilsonLoopMeasurement(
        ::Gaugefield;
        filename = "",
        printvalues = false,
        Rmax = 4,
        Tmax = 4,
        flow = false,
    )
        WL_dict = Dict{NTuple{2,Int64}, Float64}()
        for iT in 1:Tmax
            for iR in 1:Rmax
                WL_dict[iR, iT] = 0.0
            end
        end

        if printvalues
            fp = open(filename, "w")

            if flow
                @printf(fp, "%-9s\t%-7s\t%-9s", "itrj", "iflow", "tflow")
            else
                @printf(fp, "%-9s", "itrj")
            end

            for iT in 1:Tmax
                for iR in 1:Rmax
                    @printf(fp, "\t%-22s", "wilson_loop_$(iR)x$(iT)")
                end
            end

            @printf(fp, "\n")
        else
            fp = nothing
        end

        return new{typeof(fp)}(WL_dict, Tmax, Rmax, fp)
    end
end

function WilsonLoopMeasurement(U, params::WilsonLoopParameters, filename, flow=false)
    return WilsonLoopMeasurement(
        U;
        filename = filename,
        printvalues = true,
        Rmax = params.Rmax,
        Tmax = params.Tmax,
        flow = flow,
    )
end

function measure(m::WilsonLoopMeasurement{T}, U; additional_string="") where {T}
    T==IOStream && @printf(m.fp, "%-9s", additional_string)

    for iT in 1:m.Tmax
        for iR in 1:m.Rmax
            WL = tr(wilsonloop(U, iR, iT)) / (18.0*U.NV)
            m.WL_dict[iR, iT] = WL

            if T == IOStream
                @printf(m.fp, "\t%+-22.15E", WL)
            end
        end
    end

    if T == IOStream
        @printf(m.fp, "\n")
        flush(m.fp)
    end

    output = MeasurementOutput(m.WL_dict, "")
    return output
end

struct TopologicalChargeMeasurement{T} <: AbstractMeasurement
    TC_dict::Dict{String, Float64}
    fp::T

    function TopologicalChargeMeasurement(
        ::Gaugefield;
        filename = "",
        printvalues = false,
        TC_methods = ["clover"],
        flow = false,
    )
        TC_dict = Dict{String, Float64}()
        for method in TC_methods
            TC_dict[method] = 0.0
        end

        if printvalues
            fp = open(filename, "w")
            header = ""
            if flow
                header *= @sprintf("%-9s\t%-7s\t%-9s", "itrj", "iflow", "tflow")
            else
                header *= @sprintf("%-9s", "itrj")
            end

            for methodname in TC_methods
                header *= "\tQ$(rpad(methodname, 22, " "))"
            end

            println(fp, header)
        else
            fp = nothing
        end

        return new{typeof(fp)}(TC_dict, fp)
    end
end

function TopologicalChargeMeasurement(U, params::TopologicalChargeParameters, filename, flow=false)
    return TopologicalChargeMeasurement(
        U,
        filename = filename,
        printvalues = true,
        TC_methods = params.kinds_of_topological_charge,
        flow = flow,
    )
end

function measure(m::TopologicalChargeMeasurement{T}, U; additional_string="") where {T}
    measurestring = ""
    printstring = @sprintf("%-9s", additional_string)

    for methodname in keys(m.TC_dict)
        Q = top_charge(U, methodname)
        m.TC_dict[methodname] = Q
    end

    if T == IOStream
        for value in values(m.TC_dict)
            svalue = @sprintf("%+-22.15E", value)
            printstring *= "\t$svalue"
        end

        measurestring = printstring
        println(m.fp, measurestring)
        flush(m.fp)
        measurestring *= " # top_charge"
    end

    output = MeasurementOutput(m.TC_dict, measurestring)
    return output
end

# Topological charge definitions from: https://arxiv.org/pdf/1708.00696.pdf
function top_charge(U::Gaugefield, methodname::String)
    if methodname == "plaquette"
        Q = top_charge(Plaquette(), U)
    elseif methodname == "clover"
        Q = top_charge(Clover(), U)
    elseif methodname == "improved"
        Q = top_charge(Improved(), U)
    else
        error("Topological charge method '$(methodname)' not supported")
    end

    return Q
end

function top_charge(::Plaquette, U)
    @batch threadlocal=0.0::Float64 for site in eachindex(U)
        C12 = plaquette(U, 1, 2, site)
        F12 = im * traceless_antihermitian(C12)

        C13 = plaquette(U, 1, 3, site)
        F13 = im * traceless_antihermitian(C13)

        C23 = plaquette(U, 2, 3, site)
        F23 = im * traceless_antihermitian(C23)

        C14 = plaquette(U, 1, 4, site)
        F14 = im * traceless_antihermitian(C14)

        C24 = plaquette(U, 2, 4, site)
        F24 = im * traceless_antihermitian(C24)

        C34 = plaquette(U, 3, 4, site)
        F34 = im * traceless_antihermitian(C34)

        threadlocal += real(multr(F12, F34)) - real(multr(F13, F24)) + real(multr(F14, F23))
    end

    Qplaq = 1/4π^2 * sum(threadlocal)
    return Qplaq
end

function top_charge(::Clover, U)
    @batch threadlocal=0.0::Float64 for site in eachindex(U)
        C12 = clover_square(U, 1, 2, site, 1)
        F12 = im/4 * traceless_antihermitian(C12)

        C13 = clover_square(U, 1, 3, site, 1)
        F13 = im/4 * traceless_antihermitian(C13)

        C23 = clover_square(U, 2, 3, site, 1)
        F23 = im/4 * traceless_antihermitian(C23)

        C14 = clover_square(U, 1, 4, site, 1)
        F14 = im/4 * traceless_antihermitian(C14)

        C24 = clover_square(U, 2, 4, site, 1)
        F24 = im/4 * traceless_antihermitian(C24)

        C34 = clover_square(U, 3, 4, site, 1)
        F34 = im/4 * traceless_antihermitian(C34)

        threadlocal += real(multr(F12, F34)) - real(multr(F13, F24)) + real(multr(F14, F23))
    end

    Qclover = 1/4π^2 * sum(threadlocal)
    return Qclover
end

function top_charge(::Improved, U)
    Qclover = top_charge(Clover(), U)
    Qrect = top_charge_rect(U)
    Qimproved = 5/3*Qclover - 1/12*Qrect
    return Qimproved
end

function top_charge_rect(U)
    @batch threadlocal=0.0::Float64 for site in eachindex(U)
        C12 = clover_rect(U, 1, 2, site, 1, 2)
        F12 = im/8 * traceless_antihermitian(C12)

        C13 = clover_rect(U, 1, 3, site, 1, 2)
        F13 = im/8 * traceless_antihermitian(C13)

        C23 = clover_rect(U, 2, 3, site, 1, 2)
        F23 = im/8 * traceless_antihermitian(C23)

        C14 = clover_rect(U, 1, 4, site, 1, 2)
        F14 = im/8 * traceless_antihermitian(C14)

        C24 = clover_rect(U, 2, 4, site, 1, 2)
        F24 = im/8 * traceless_antihermitian(C24)

        C34 = clover_rect(U, 3, 4, site, 1, 2)
        F34 = im/8 * traceless_antihermitian(C34)

        threadlocal += real(multr(F12, F34)) - real(multr(F13, F24)) + real(multr(F14, F23))
    end

    Qrect = 2/4π^2 * sum(threadlocal)
    return Qrect
end

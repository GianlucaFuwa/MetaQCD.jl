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
    return TopologicalChargeMeasurement(U, filename = filename, printvalues = true,
                                        TC_methods = params.kinds_of_topological_charge,
                                        flow = flow)
end

function measure(m::TopologicalChargeMeasurement{T}, U; additional_string="") where {T}
    measurestring = ""
    printstring = @sprintf("%-9s", additional_string)

    for methodname in keys(m.TC_dict)
        Q = top_charge(U, methodname)
        m.TC_dict[methodname] = Q
    end

    if T ≡ IOStream
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
    out = zeros(Float64, 8nthreads())

    @threads for site in eachindex(U)
        out[8threadid()] += top_charge_density_plaq(U, site)
    end

    Q_plaq = 1/4π^2 * sum(out)
    return Q_plaq
end

function top_charge(::Clover, U)
    out = zeros(Float64, 8nthreads())

    @threads for site in eachindex(U)
        out[8threadid()] += top_charge_density_clover(U, site)
    end

    Q_clover = 1/4π^2 * sum(out)
    return Q_clover
end

function top_charge(::Improved, U)
    out = zeros(Float64, 8nthreads())
    c₁ = 5/3
    c₂ = -2/12

    @threads for site in eachindex(U)
        out[8threadid()] += top_charge_density_imp(U, site, c₁, c₂)
    end

    Q_imp = 1/4π^2 * sum(out)
    return Q_imp
end

function top_charge_density_plaq(U, site)
    C₁₂ = plaquette(U, 1, 2, site)
    F₁₂ = im * traceless_antihermitian(C₁₂)
    C₁₃ = plaquette(U, 1, 3, site)
    F₁₃ = im * traceless_antihermitian(C₁₃)
    C₂₃ = plaquette(U, 2, 3, site)
    F₂₃ = im * traceless_antihermitian(C₂₃)
    C₁₄ = plaquette(U, 1, 4, site)
    F₁₄ = im * traceless_antihermitian(C₁₄)
    C₂₄ = plaquette(U, 2, 4, site)
    F₂₄ = im * traceless_antihermitian(C₂₄)
    C₃₄ = plaquette(U, 3, 4, site)
    F₃₄ = im * traceless_antihermitian(C₃₄)

    qₙ = real(multr(F₁₂, F₃₄)) - real(multr(F₁₃, F₂₄)) + real(multr(F₁₄, F₂₃))
    return qₙ
end

function top_charge_density_clover(U, site)
    C₁₂ = clover_square(U, 1, 2, site, 1)
    F₁₂ = im/4 * traceless_antihermitian(C₁₂)
    C₁₃ = clover_square(U, 1, 3, site, 1)
    F₁₃ = im/4 * traceless_antihermitian(C₁₃)
    C₂₃ = clover_square(U, 2, 3, site, 1)
    F₂₃ = im/4 * traceless_antihermitian(C₂₃)
    C₁₄ = clover_square(U, 1, 4, site, 1)
    F₁₄ = im/4 * traceless_antihermitian(C₁₄)
    C₂₄ = clover_square(U, 2, 4, site, 1)
    F₂₄ = im/4 * traceless_antihermitian(C₂₄)
    C₃₄ = clover_square(U, 3, 4, site, 1)
    F₃₄ = im/4 * traceless_antihermitian(C₃₄)

    out = real(multr(F₁₂, F₃₄)) - real(multr(F₁₃, F₂₄)) + real(multr(F₁₄, F₂₃))
    return out
end

function top_charge_density_imp(U, site, c₁, c₂)
    q_clov = top_charge_density_clover(U, site)
    q_rect = top_charge_density_rect(U, site)
    q_imp = c₁*q_clov - c₂*q_rect
    return q_imp
end

function top_charge_density_rect(U, site)
    C₁₂ = clover_rect(U, 1, 2, site, 1, 2)
    F₁₂ = im/8 * traceless_antihermitian(C₁₂)
    C₁₃ = clover_rect(U, 1, 3, site, 1, 2)
    F₁₃ = im/8 * traceless_antihermitian(C₁₃)
    C₂₃ = clover_rect(U, 2, 3, site, 1, 2)
    F₂₃ = im/8 * traceless_antihermitian(C₂₃)
    C₁₄ = clover_rect(U, 1, 4, site, 1, 2)
    F₁₄ = im/8 * traceless_antihermitian(C₁₄)
    C₂₄ = clover_rect(U, 2, 4, site, 1, 2)
    F₂₄ = im/8 * traceless_antihermitian(C₂₄)
    C₃₄ = clover_rect(U, 3, 4, site, 1, 2)
    F₃₄ = im/8 * traceless_antihermitian(C₃₄)

    out = real(multr(F₁₂, F₃₄)) - real(multr(F₁₃, F₂₄)) + real(multr(F₁₄, F₂₃))
    return out
end

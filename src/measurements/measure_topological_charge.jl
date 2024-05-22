struct TopologicalChargeMeasurement{T} <: AbstractMeasurement
    TC_dict::Dict{String,Float64} # topological charge definition => value
    fp::T # file pointer
    function TopologicalChargeMeasurement(
        ::Gaugefield; filename="", printvalues=false, TC_methods=["clover"], flow=false
    )
        TC_dict = Dict{String,Float64}()
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

function TopologicalChargeMeasurement(
    U, params::TopologicalChargeParameters, filename, flow=false
)
    return TopologicalChargeMeasurement(
        U;
        filename=filename,
        printvalues=true,
        TC_methods=params.kinds_of_topological_charge,
        flow=flow,
    )
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

function top_charge(::Plaquette, U::Gaugefield{CPU})
    Q = 0.0

    @batch reduction = (+, Q) for site in eachindex(U)
        Q += top_charge_density_plaq(U, site)
    end

    return Q / 4π^2
end

function top_charge(::Clover, U::Gaugefield{CPU})
    Q = 0.0

    @batch reduction = (+, Q) for site in eachindex(U)
        Q += top_charge_density_clover(U, site)
    end

    return Q / 4π^2
end

function top_charge(::Improved, U::Gaugefield{CPU})
    Q = 0.0
    c₀ = float_type(U)(5 / 3)
    c₁ = float_type(U)(-2 / 12)

    @batch reduction = (+, Q) for site in eachindex(U)
        Q += top_charge_density_imp(U, site, c₀, c₁)
    end

    return Q / 4π^2
end

function top_charge_density_plaq(U, site)
    C₁₂ = plaquette(U, 1i32, 2i32, site)
    F₁₂ = im * antihermitian(C₁₂)
    C₁₃ = plaquette(U, 1i32, 3i32, site)
    F₁₃ = im * antihermitian(C₁₃)
    C₂₃ = plaquette(U, 2i32, 3i32, site)
    F₂₃ = im * antihermitian(C₂₃)
    C₁₄ = plaquette(U, 1i32, 4i32, site)
    F₁₄ = im * antihermitian(C₁₄)
    C₂₄ = plaquette(U, 2i32, 4i32, site)
    F₂₄ = im * antihermitian(C₂₄)
    C₃₄ = plaquette(U, 3i32, 4i32, site)
    F₃₄ = im * antihermitian(C₃₄)

    qₙ = real(multr(F₁₂, F₃₄)) - real(multr(F₁₃, F₂₄)) + real(multr(F₁₄, F₂₃))
    return qₙ
end

function top_charge_density_clover(U, site)
    C₁₂ = clover_square(U, 1i32, 2i32, site, 1i32)
    F₁₂ = im * 1//4 * antihermitian(C₁₂)
    C₁₃ = clover_square(U, 1i32, 3i32, site, 1i32)
    F₁₃ = im * 1//4 * antihermitian(C₁₃)
    C₂₃ = clover_square(U, 2i32, 3i32, site, 1i32)
    F₂₃ = im * 1//4 * antihermitian(C₂₃)
    C₁₄ = clover_square(U, 1i32, 4i32, site, 1i32)
    F₁₄ = im * 1//4 * antihermitian(C₁₄)
    C₂₄ = clover_square(U, 2i32, 4i32, site, 1i32)
    F₂₄ = im * 1//4 * antihermitian(C₂₄)
    C₃₄ = clover_square(U, 3i32, 4i32, site, 1i32)
    F₃₄ = im * 1//4 * antihermitian(C₃₄)

    out = real(multr(F₁₂, F₃₄)) - real(multr(F₁₃, F₂₄)) + real(multr(F₁₄, F₂₃))
    return out
end

function top_charge_density_imp(U, site, c₀, c₁)
    q_clov = top_charge_density_clover(U, site)
    q_rect = top_charge_density_rect(U, site)
    q_imp = c₀ * q_clov + c₁ * q_rect
    return q_imp
end

function top_charge_density_rect(U, site)
    C₁₂ = clover_rect(U, 1i32, 2i32, site, 1i32, 2i32)
    F₁₂ = im * 1//8 * antihermitian(C₁₂)
    C₁₃ = clover_rect(U, 1i32, 3i32, site, 1i32, 2i32)
    F₁₃ = im * 1//8 * antihermitian(C₁₃)
    C₂₃ = clover_rect(U, 2i32, 3i32, site, 1i32, 2i32)
    F₂₃ = im * 1//8 * antihermitian(C₂₃)
    C₁₄ = clover_rect(U, 1i32, 4i32, site, 1i32, 2i32)
    F₁₄ = im * 1//8 * antihermitian(C₁₄)
    C₂₄ = clover_rect(U, 2i32, 4i32, site, 1i32, 2i32)
    F₂₄ = im * 1//8 * antihermitian(C₂₄)
    C₃₄ = clover_rect(U, 3i32, 4i32, site, 1i32, 2i32)
    F₃₄ = im * 1//8 * antihermitian(C₃₄)

    out = real(multr(F₁₂, F₃₄)) - real(multr(F₁₃, F₂₄)) + real(multr(F₁₄, F₂₃))
    return out
end

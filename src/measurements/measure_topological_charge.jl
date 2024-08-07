struct TopologicalChargeMeasurement{T} <: AbstractMeasurement
    TC_dict::Dict{String,Float64} # topological charge definition => value
    filename::T
    function TopologicalChargeMeasurement(
        ::Gaugefield; filename="", TC_methods=["clover"], flow=false
    )
        TC_dict = Dict{String,Float64}()
        for method in TC_methods
            @level1("|    Method: $(method)")
            if method == "plaquette"
                TC_dict["plaquette"] = 0.0
            elseif method == "clover"
                TC_dict["clover"] = 0.0
            elseif method == "improved"
                TC_dict["improved"] = 0.0
            else
                error("Topological charge method $method not supported")
            end
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

            for method in keys(TC_dict)
                header *= @sprintf("%-25s", "Q_$(method)")
            end

            open(path, "w") do fp
                println(fp, header)
            end
        else
            rpath = nothing
        end

        T = typeof(rpath)
        return new{T}(TC_dict, rpath)
    end
end

function TopologicalChargeMeasurement(
    U, params::TopologicalChargeParameters, filename, flow=false
)
    return TopologicalChargeMeasurement(
        U;
        filename=filename,
        TC_methods=params.kinds_of_topological_charge,
        flow=flow,
    )
end

@inline function measure(
    m::TopologicalChargeMeasurement{Nothing}, U, ::Integer, itrj, flow=nothing,
)
    TC_dict = m.TC_dict
    iflow, _ = isnothing(flow) ? (0, 0.0) : flow

    for method in keys(TC_dict)
        Q = top_charge(U, method)
        TC_dict[method] = Q

        if !isnothing(flow)
            @level1("$itrj\t$Q # topcharge_$(method)_flow_$(iflow)")
        else
            @level1("$itrj\t$Q # topcharge_$(method)")
        end
    end

    return TC_dict
end

function measure(
    m::TopologicalChargeMeasurement{T}, U, myinstance, itrj, flow=nothing,
) where {T<:AbstractString}
    TC_dict = m.TC_dict
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    for method in keys(TC_dict)
        TC_dict[method] = top_charge(U, method)
    end

    if T !== Nothing
        filename = set_ext!(m.filename, myinstance)
        fp = fopen(filename, "a")
        printf(fp, "%-11i", itrj)

        if !isnothing(flow)
            printf(fp, "%-7i", iflow)
            printf(fp, "%-9.5f", τ)
        end

        for method in keys(TC_dict)
            v = TC_dict[method]
            printf(fp, "%+-25.15E", v)
        end

        printf(fp, "\n")
        fclose(fp)
    end

    return TC_dict
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

    return Q/4π^2
end

function top_charge(::Clover, U::Gaugefield{CPU,T}) where {T}
    Q = 0.0

    @batch reduction = (+, Q) for site in eachindex(U)
        Q += top_charge_density_clover(U, site, T)
    end

    return Q/4π^2
end

function top_charge(::Improved, U::Gaugefield{CPU,T}) where {T}
    Q = 0.0
    c₀ = T(5/3)
    c₁ = T(-2/12)

    @batch reduction = (+, Q) for site in eachindex(U)
        Q += top_charge_density_imp(U, site, c₀, c₁, T)
    end

    return Q/4π^2
end

function top_charge_density_plaq(U, site)
    C₁₂ = plaquette(U, 1i32, 2i32, site)
    F₁₂ = im * traceless_antihermitian(C₁₂)
    C₁₃ = plaquette(U, 1i32, 3i32, site)
    F₁₃ = im * traceless_antihermitian(C₁₃)
    C₂₃ = plaquette(U, 2i32, 3i32, site)
    F₂₃ = im * traceless_antihermitian(C₂₃)
    C₁₄ = plaquette(U, 1i32, 4i32, site)
    F₁₄ = im * traceless_antihermitian(C₁₄)
    C₂₄ = plaquette(U, 2i32, 4i32, site)
    F₂₄ = im * traceless_antihermitian(C₂₄)
    C₃₄ = plaquette(U, 3i32, 4i32, site)
    F₃₄ = im * traceless_antihermitian(C₃₄)

    qₙ = real(multr(F₁₂, F₃₄)) - real(multr(F₁₃, F₂₄)) + real(multr(F₁₄, F₂₃))
    return qₙ
end

function top_charge_density_clover(U, site, ::Type{T}) where {T}
    C₁₂ = clover_square(U, 1i32, 2i32, site, 1i32)
    F₁₂ = im * T(1/4) * traceless_antihermitian(C₁₂)
    C₁₃ = clover_square(U, 1i32, 3i32, site, 1i32)
    F₁₃ = im * T(1/4) * traceless_antihermitian(C₁₃)
    C₂₃ = clover_square(U, 2i32, 3i32, site, 1i32)
    F₂₃ = im * T(1/4) * traceless_antihermitian(C₂₃)
    C₁₄ = clover_square(U, 1i32, 4i32, site, 1i32)
    F₁₄ = im * T(1/4) * traceless_antihermitian(C₁₄)
    C₂₄ = clover_square(U, 2i32, 4i32, site, 1i32)
    F₂₄ = im * T(1/4) * traceless_antihermitian(C₂₄)
    C₃₄ = clover_square(U, 3i32, 4i32, site, 1i32)
    F₃₄ = im * T(1/4) * traceless_antihermitian(C₃₄)

    out = real(multr(F₁₂, F₃₄)) - real(multr(F₁₃, F₂₄)) + real(multr(F₁₄, F₂₃))
    return out
end

function top_charge_density_imp(U, site, c₀, c₁, ::Type{T}) where {T}
    q_clov = top_charge_density_clover(U, site, T)
    q_rect = top_charge_density_rect(U, site, T)
    q_imp = c₀ * q_clov + c₁ * q_rect
    return q_imp
end

function top_charge_density_rect(U, site, ::Type{T}) where {T}
    C₁₂ = clover_rect(U, 1i32, 2i32, site, 1i32, 2i32)
    F₁₂ = im * T(1/8) * traceless_antihermitian(C₁₂)
    C₁₃ = clover_rect(U, 1i32, 3i32, site, 1i32, 2i32)
    F₁₃ = im * T(1/8) * traceless_antihermitian(C₁₃)
    C₂₃ = clover_rect(U, 2i32, 3i32, site, 1i32, 2i32)
    F₂₃ = im * T(1/8) * traceless_antihermitian(C₂₃)
    C₁₄ = clover_rect(U, 1i32, 4i32, site, 1i32, 2i32)
    F₁₄ = im * T(1/8) * traceless_antihermitian(C₁₄)
    C₂₄ = clover_rect(U, 2i32, 4i32, site, 1i32, 2i32)
    F₂₄ = im * T(1/8) * traceless_antihermitian(C₂₄)
    C₃₄ = clover_rect(U, 3i32, 4i32, site, 1i32, 2i32)
    F₃₄ = im * T(1/8) * traceless_antihermitian(C₃₄)

    out = real(multr(F₁₂, F₃₄)) - real(multr(F₁₃, F₂₄)) + real(multr(F₁₄, F₂₃))
    return out
end

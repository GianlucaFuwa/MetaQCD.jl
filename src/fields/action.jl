abstract type AbstractGaugeAction end

struct WilsonGaugeAction <: AbstractGaugeAction end
struct SymanzikTreeGaugeAction <: AbstractGaugeAction end
struct SymanzikTadGaugeAction <: AbstractGaugeAction end
struct IwasakiGaugeAction <: AbstractGaugeAction end
struct DBW2GaugeAction <: AbstractGaugeAction end

const GAUGE_ACTION = Dict{String,Type{<:AbstractGaugeAction}}(
    "wilson" => WilsonGaugeAction,
    "symanzik_tree" => SymanzikTreeGaugeAction,
    "iwasaki" => IwasakiGaugeAction,
    "dbw2" => DBW2GaugeAction,
)

stencil_size(::Type{WilsonGaugeAction}) = 1
stencil_size(::Type{SymanzikTreeGaugeAction}) = 2
stencil_size(::Type{SymanzikTadGaugeAction}) = 2
stencil_size(::Type{IwasakiGaugeAction}) = 2
stencil_size(::Type{DBW2GaugeAction}) = 2

function calc_gauge_action(U::Gaugefield, methodname::String)
    if methodname == "wilson"
        Sg = calc_gauge_action(WilsonGaugeAction(), U)
    elseif methodname == "symanzik_tree"
        Sg = calc_gauge_action(SymanzikTreeGaugeAction(), U)
    elseif methodname == "symanzik_tad"
        Sg = calc_gauge_action(SymanzikTadGaugeAction(), U)
    elseif methodname == "iwasaki"
        Sg = calc_gauge_action(IwasakiGaugeAction(), U)
    elseif methodname == "dbw2"
        Sg = calc_gauge_action(DBW2GaugeAction(), U)
    else
        error("method $methodname is not supported in gauge action measurement")
    end

    return Sg
end

function calc_gauge_action(U::Gaugefield{B,T,M,GA}) where {B,T,M,GA}
    # Uhigh = convert_field(B, U, Float64)
    return calc_gauge_action(GA(), U)
end

function calc_gauge_action(::WilsonGaugeAction, U::Gaugefield)
    P = plaquette_trace_sum(U)
    Sg_wilson = U.β * (6 * length(U) - 1 / 3 * P)
    return Sg_wilson
end

function calc_gauge_action(::SymanzikTreeGaugeAction, U::Gaugefield)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    Sg_plaq = 6 * length(U) - 1 / 3 * P
    Sg_rect = 12 * length(U) - 1 / 3 * R
    Sg_symanzik = U.β * ((1 + 8 / 12) * Sg_plaq - 1 / 12 * Sg_rect)
    return Sg_symanzik
end

function calc_gauge_action(::SymanzikTadGaugeAction, U::Gaugefield)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    u0sq = sqrt(1 / (6 * length(U) * 3) * P)
    Sg_plaq = 6 * length(U) - 1 / 3 * P
    Sg_rect = 12 * length(U) - 1 / 3 * R
    Sg_symanzik = U.β * ((1 + 8 / 12) * Sg_plaq - 1 / 12u0sq * Sg_rect)
    return Sg_symanzik
end

function calc_gauge_action(::IwasakiGaugeAction, U::Gaugefield)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    Sg_plaq = 6 * length(U) - 1 / 3 * P
    Sg_rect = 12 * length(U) - 1 / 3 * R
    Sg_iwasaki = U.β * ((1 + 8 * 0.331) * Sg_plaq - 0.331 * Sg_rect)
    return Sg_iwasaki
end

function calc_gauge_action(::DBW2GaugeAction, U::Gaugefield)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    Sg_plaq = 6 * length(U) - 1 / 3 * P
    Sg_rect = 12 * length(U) - 1 / 3 * R
    Sg_dbw2 = U.β * ((1 + 8 * 1.409) * Sg_plaq - 1.409 * Sg_rect)
    return Sg_dbw2
end

function gauge_action_deriv!(
    dU::Colorfield{B,T}, staples::Colorfield{B,T}, U::Gaugefield{B,TU,M}, fac=1
) where {B,T,M,TU}
    mβover6 = T(-U.β*fac / 6)
    gaction = gauge_action(U)()
    itr = eachindex(dU, staples, U)

    parallelfor(itr, B, Val(M), (U,), (dU, staples), (U, dU, staples)) do site, (U, dU, staples)
        for μ in 1:4
            A = staple(gaction, U, μ, site)
            staples[μ, site] = A
            UA = cmatmul_od(U[μ, site], A)
            dU[μ, site] = mβover6 * traceless_antihermitian(UA)
        end
    end

    return nothing
end

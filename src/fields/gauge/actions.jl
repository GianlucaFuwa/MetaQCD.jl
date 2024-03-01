abstract type AbstractGaugeAction end

struct WilsonGaugeAction <: AbstractGaugeAction end
struct SymanzikTreeGaugeAction <: AbstractGaugeAction end
struct SymanzikTadGaugeAction <: AbstractGaugeAction end
struct IwasakiGaugeAction <: AbstractGaugeAction end
struct DBW2GaugeAction <: AbstractGaugeAction end

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

calc_gauge_action(U::Gaugefield{B,T,A,GA}) where {B,T,A,GA} = calc_gauge_action(GA(), U)

function calc_gauge_action(::WilsonGaugeAction, U)
    P = plaquette_trace_sum(U)
    Sg_wilson = U.β * (6*U.NV - 1/3*P)
    return Sg_wilson
end

function calc_gauge_action(::SymanzikTreeGaugeAction, U)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    Sg_plaq = 6*U.NV - 1/3*P
    Sg_rect = 12*U.NV - 1/3*R
    Sg_symanzik = U.β * ((1 + 8/12)*Sg_plaq - 1/12*Sg_rect)
    return Sg_symanzik
end

function calc_gauge_action(::SymanzikTadGaugeAction, U)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    u0sq = sqrt(1/(6*U.NV*U.NC) * P)
    Sg_plaq = 6*U.NV - 1/3*P
    Sg_rect = 12*U.NV - 1/3*R
    Sg_symanzik = U.β * ((1 + 8/12)*Sg_plaq - 1/12u0sq*Sg_rect)
    return Sg_symanzik
end

function calc_gauge_action(::IwasakiGaugeAction, U)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    Sg_plaq = 6*U.NV - 1/3*P
    Sg_rect = 12*U.NV - 1/3*R
    Sg_iwasaki = U.β * ((1 + 8*0.331)*Sg_plaq - 0.331*Sg_rect)
    return Sg_iwasaki
end

function calc_gauge_action(::DBW2GaugeAction, U)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    Sg_plaq = 6*U.NV - 1/3*P
    Sg_rect = 12*U.NV - 1/3*R
    Sg_dbw2 = U.β * ((1 + 8*1.409) * Sg_plaq - 1.409*Sg_rect)
    return Sg_dbw2
end

function plaquette_trace_sum(U::Gaugefield{CPU,T}) where {T}
    P = zero(T)

    @batch reduction=(+, P) for site in eachindex(U)
        for μ in 1:3
            for ν in μ+1:4
                P += real(tr(plaquette(U, μ, ν, site)))
            end
        end
    end

    return P
end

function rect_trace_sum(U::Gaugefield{CPU,T}) where {T}
    R = zero(T)

    @batch reduction=(+, R) for site in eachindex(U)
        for μ in 1:3
            for ν in μ+1:4
                R += real(tr(rect_1x2(U, μ, ν, site))) + real(tr(rect_2x1(U, μ, ν, site)))
            end
        end
    end

    return R
end

function plaquette(U, μ, ν, site)
    Nμ = dims(U)[μ]
    Nν = dims(U)[ν]
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteν⁺ = move(site, ν, 1, Nν)
    plaq = cmatmul_oodd(U[μ,site], U[ν,siteμ⁺], U[μ,siteν⁺], U[ν,site])
    return plaq
end

function rect_2x1(U, μ, ν, site)
    Nμ = dims(U)[μ]
    Nν = dims(U)[ν]
    siteμ⁺   = move(site, μ, 1, Nμ)
    siteμ²⁺  = move(siteμ⁺, μ, 1, Nμ)
    siteμ⁺ν⁺ = move(siteμ⁺, ν, 1, Nν)
    siteν⁺   = move(site, ν, 1, Nν)
    plaq = cmatmul_oo(cmatmul_oood(U[μ,site], U[μ,siteμ⁺], U[ν,siteμ²⁺], U[μ,siteμ⁺ν⁺]),
                      cmatmul_dd(U[μ,siteν⁺], U[ν,site]))
    return plaq
end

function rect_1x2(U, μ, ν, site)
    Nμ = dims(U)[μ]
    Nν = dims(U)[ν]
    siteμ⁺   = move(site, μ, 1, Nμ)
    siteμ⁺ν⁺ = move(siteμ⁺, ν, 1, Nν)
    siteν⁺   = move(site, ν, 1, Nν)
    siteν²⁺  = move(siteν⁺, ν, 1, Nν)
    plaq = cmatmul_oo(cmatmul_oood(U[μ,site], U[ν,siteμ⁺], U[ν,siteμ⁺ν⁺], U[μ,siteν²⁺]),
                      cmatmul_dd(U[ν,siteν⁺], U[ν,site]))
    return plaq
end

function plaquette_2x2(U, μ, ν, site)
    Nμ = dims(U)[μ]
    Nν = dims(U)[ν]
    siteμ⁺    = move(site, μ, 1, Nμ)
    siteμ²⁺   = move(siteμ⁺, μ, 1, Nμ)
    siteμ²⁺ν⁺ = move(siteμ²⁺, ν, 1, Nν)
    siteν⁺    = move(site, ν, 1, Nν)
    siteν²⁺   = move(siteν⁺, ν, 1, Nν)
    siteμ⁺ν²⁺ = move(siteν²⁺, μ, 1, Nμ)
    plaq = cmatmul_oo(cmatmul_oooo(U[μ,site], U[μ,siteμ⁺], U[ν,siteμ²⁺], U[ν,siteμ²⁺ν⁺]),
                      cmatmul_dddd(U[μ,siteμ⁺ν²⁺], U[μ,siteν²⁺], U[ν,siteν⁺], U[ν,site]))
    return plaq
end

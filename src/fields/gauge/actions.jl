struct WilsonGaugeAction <: AbstractGaugeAction end
struct SymanzikTreeGaugeAction <: AbstractGaugeAction end
struct SymanzikTadGaugeAction <: AbstractGaugeAction end
struct IwasakiGaugeAction <: AbstractGaugeAction end
struct DBW2GaugeAction <: AbstractGaugeAction end

calc_gauge_action(U::Gaugefield{GA}) where {GA} = calc_gauge_action(GA(), U)

function calc_gauge_action(::WilsonGaugeAction, U)
    P = plaquette_trace_sum(U)
    Sg_wilson = U.β * (6 * U.NV - 1/3 * P)
    return Sg_wilson
end

function calc_gauge_action(::SymanzikTreeGaugeAction, U)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    Sg_plaq = 6 * U.NV - 1/3 * P
    Sg_rect = 12 * U.NV - 1/3 * R
    Sg_symanzik = U.β * ((1 + 8/12) * Sg_plaq - 1/12 * Sg_rect)
    return Sg_symanzik
end

function calc_gauge_action(::SymanzikTadGaugeAction, U)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    u0sq = sqrt(1/(6 * U.NV * U.NC) * P)
    Sg_plaq = 6 * U.NV - 1/3 * P
    Sg_rect = 12 * U.NV - 1/3 * R
    Sg_symanzik = U.β * ((1 + 8/12) * Sg_plaq - 1/12u0sq * Sg_rect)
    return Sg_symanzik
end

function calc_gauge_action(::IwasakiGaugeAction, U)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    Sg_plaq = 6 * U.NV - 1/3 * P
    Sg_rect = 12 * U.NV - 1/3 * R
    Sg_iwasaki = U.β * ((1 + 8*0.331) * Sg_plaq - 0.331 * Sg_rect)
    return Sg_iwasaki
end

function calc_gauge_action(::DBW2GaugeAction, U)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    Sg_plaq = 6 * U.NV - 1/3 * P
    Sg_rect = 12 * U.NV - 1/3 * R
    Sg_dbw2 = U.β * ((1 + 8 * 1.409) * Sg_plaq - 1.409 * Sg_rect)
    return Sg_dbw2
end

function plaquette_trace_sum(U)
    @batch threadlocal=0.0::Float64 for site in eachindex(U)
        for μ in 1:3
            for ν in μ+1:4
                threadlocal += real(tr(plaquette(U, μ, ν, site)))
            end
        end
    end

    return sum(threadlocal)
end

function rect_trace_sum(U)
    @batch threadlocal=0.0::Float64 for site in eachindex(U)
        for μ in 1:3
            for ν in μ+1:4
                threadlocal +=
                    real(tr(rect_1x2(U, μ, ν, site))) +
                    real(tr(rect_2x1(U, μ, ν, site)))
            end
        end
    end

    return sum(threadlocal)
end

function plaquette(U, μ, ν, site)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    siteμp = move(site, μ, 1, Nμ)
    siteνp = move(site, ν, 1, Nν)
    plaq = cmatmul_oodd(U[μ][site], U[ν][siteμp], U[μ][siteνp], U[ν][site])
    return plaq
end

function rect_2x1(U, μ, ν, site)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    siteμp = move(site, μ, 1, Nμ)
    site2μp = move(siteμp, μ, 1, Nμ)
    siteμpνp = move(siteμp, ν, 1, Nν)
    siteνp = move(site, ν, 1, Nν)
    plaq = cmatmul_oo(
        cmatmul_oood(U[μ][site], U[μ][siteμp], U[ν][site2μp], U[μ][siteμpνp]),
        cmatmul_dd(U[μ][siteνp], U[ν][site]),
    )
    return plaq
end

function rect_1x2(U, μ, ν, site)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    siteμp = move(site, μ, 1, Nμ)
    siteμpνp = move(siteμp, ν, 1, Nν)
    siteνp = move(site, ν, 1, Nν)
    site2νp = move(siteνp, ν, 1, Nν)
    plaq = cmatmul_oo(
        cmatmul_oood(U[μ][site], U[ν][siteμp], U[ν][siteμpνp], U[μ][site2νp]),
        cmatmul_dd(U[ν][siteνp], U[ν][site]),
    )
    return plaq
end

function plaquette_2x2(U, μ, ν, site)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    siteμp = move(site, μ, 1, Nμ)
    site2μp = move(siteμp, μ, 1, Nμ)
    site2μpνp = move(site2μp, ν, 1, Nν)
    siteνp = move(site, ν, 1, Nν)
    site2νp = move(siteνp, ν, 1, Nν)
    siteμp2νp =  move(site2νp, μ, 1, Nμ)
    plaq = cmatmul_oo(
        cmatmul_oooo(U[μ][site], U[μ][siteμp], U[ν][site2μp], U[ν][site2μpνp]),
        cmatmul_dddd(U[μ][siteμp2νp], U[μ][site2νp], U[ν][siteνp], U[ν][site]),
    )
    return plaq
end

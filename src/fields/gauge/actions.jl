struct WilsonGaugeAction <: AbstractGaugeAction end
struct SymanzikTreeGaugeAction <: AbstractGaugeAction end
struct SymanzikTadGaugeAction <: AbstractGaugeAction end
struct IwasakiGaugeAction <: AbstractGaugeAction end
struct DBW2GaugeAction <: AbstractGaugeAction end

function calc_gauge_action(u::Gaugefield{T}) where {T}
    return T()(u)
end

function (::WilsonGaugeAction)(u::Gaugefield)
    P = plaquette_trace_sum(u)
    Sg_wilson = u.β * (6 * u.NV - 1/3 * P)
    return Sg_wilson
end

function (::SymanzikTreeGaugeAction)(u::Gaugefield)
    P = plaquette_trace_sum(u)
    R = rect_trace_sum(u)
    Sg_plaq = 6 * u.NV - 1/3 * P
    Sg_rect = 12 * u.NV - 1/3 * R
    Sg_symanzik = u.β * ((1 + 8/12) * Sg_plaq - 1/12 * Sg_rect) 
    return Sg_symanzik
end

function (::SymanzikTadGaugeAction)(u::Gaugefield)
    P = plaquette_trace_sum(u)
    R = rect_trace_sum(u)
    u0sq = sqrt(1/3 * P)
    Sg_plaq = 6 * u.NV - 1/3 * P
    Sg_rect = 12 * u.NV - 1/3 * R
    Sg_symanzik = u.β * ((1 + 8/12) * Sg_plaq - 1/12u0sq * Sg_rect) 
    return Sg_symanzik
end

function (::IwasakiGaugeAction)(u::Gaugefield)
    P = plaquette_trace_sum(u)
    R = rect_trace_sum(u)
    Sg_plaq = 6 * u.NV - 1/3 * P
    Sg_rect = 12 * u.NV - 1/3 * R
    Sg_iwasaki = u.β * ((1 + 8*0.331) * Sg_plaq - 0.331 * Sg_rect) 
    return Sg_iwasaki
end

function (::DBW2GaugeAction)(u::Gaugefield)
    P = plaquette_trace_sum(u)
    R = rect_trace_sum(u)
    Sg_plaq = 6 * u.NV - 1/3 * P
    Sg_rect = 12 * u.NV - 1/3 * R
    Sg_dbw2 = u.β * ((1 + 8 * 1.409) * Sg_plaq - 1.409 * Sg_rect) 
    return Sg_dbw2
end

function plaquette(u::Gaugefield, μ, ν, site::SiteCoords)
    Nμ = size(u)[μ]
    Nν = size(u)[ν]
    siteμ = move(site, μ, 1, Nμ)
    siteν = move(site, ν, 1, Nν)
    plaq = u[μ][site] * u[ν][siteμ] * u[μ][siteν]' * u[ν][site]'
    return plaq
end

function plaquette_trace_sum(u::Gaugefield)
    plaq = 0.0
    NX, NY, NZ, NT = size(u)

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)
                    for μ in 1:3
                        for ν in μ+1:4
                            plaq += 
                                real(tr(plaquette(u, μ, ν, site)))
                        end
                    end  
                end	
            end
        end
    end

    return plaq
end

function rect_trace_sum(u::Gaugefield)
    rect = 0.0
    NX, NY, NZ, NT = size(u)

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)
                    for μ in 1:3
                        for ν in μ+1:4
                            rect +=
                                real(tr(wilsonloop_top_right(u, μ, ν, site, 1, 2))) +
                                real(tr(wilsonloop_top_right(u, μ, ν, site, 2, 1)))
                        end
                    end 
                end
            end
        end
    end

    return rect
end
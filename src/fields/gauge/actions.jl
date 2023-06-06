struct WilsonGaugeAction <: AbstractGaugeAction end
struct SymanzikTreeGaugeAction <: AbstractGaugeAction end
struct SymanzikTadGaugeAction <: AbstractGaugeAction end
struct IwasakiGaugeAction <: AbstractGaugeAction end
struct DBW2GaugeAction <: AbstractGaugeAction end

function calc_gauge_action(U::Gaugefield{T}) where {T}
    return T()(U)
end

function (::WilsonGaugeAction)(U::T) where {T<:Gaugefield}
    P = plaquette_trace_sum(U)
    Sg_wilson = U.β * (6 * U.NV - 1/3 * P)
    return Sg_wilson
end

function (::SymanzikTreeGaugeAction)(U::T) where {T<:Gaugefield}
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    Sg_plaq = 6 * U.NV - 1/3 * P
    Sg_rect = 12 * U.NV - 1/3 * R
    Sg_symanzik = U.β * ((1 + 8/12) * Sg_plaq - 1/12 * Sg_rect) 
    return Sg_symanzik
end

function (::SymanzikTadGaugeAction)(U::T) where {T<:Gaugefield}
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    u0sq = sqrt(1/3 * P)
    Sg_plaq = 6 * U.NV - 1/3 * P
    Sg_rect = 12 * U.NV - 1/3 * R
    Sg_symanzik = U.β * ((1 + 8/12) * Sg_plaq - 1/12u0sq * Sg_rect) 
    return Sg_symanzik
end

function (::IwasakiGaugeAction)(U::T) where {T<:Gaugefield}
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    Sg_plaq = 6 * U.NV - 1/3 * P
    Sg_rect = 12 * U.NV - 1/3 * R
    Sg_iwasaki = U.β * ((1 + 8*0.331) * Sg_plaq - 0.331 * Sg_rect) 
    return Sg_iwasaki
end

function (::DBW2GaugeAction)(U::T) where {T<:Gaugefield}
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    Sg_plaq = 6 * U.NV - 1/3 * P
    Sg_rect = 12 * U.NV - 1/3 * R
    Sg_dbw2 = U.β * ((1 + 8 * 1.409) * Sg_plaq - 1.409 * Sg_rect) 
    return Sg_dbw2
end

function plaquette(U::T, μ, ν, site::SiteCoords) where {T<:Gaugefield}
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    siteμ = move(site, μ, 1, Nμ)
    siteν = move(site, ν, 1, Nν)
    plaq = U[μ][site] * U[ν][siteμ] * U[μ][siteν]' * U[ν][site]'
    return plaq
end

function plaquette_trace_sum(U::T) where {T<:Gaugefield}
    NX, NY, NZ, NT = size(U)
    spacing = 8
    plaq = zeros(Float64, nthreads() * spacing)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    for μ in 1:3
                        for ν in μ+1:4
                            plaq[threadid() * spacing] += 
                                real(tr(plaquette(U, μ, ν, site)))
                        end
                    end

                end	
            end
        end
    end

    return sum(plaq)
end

function rect_trace_sum(U::T) where {T<:Gaugefield}
    NX, NY, NZ, NT = size(U)
    spacing = 8
    rect = zeros(Float64, nthreads() * spacing)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    for μ in 1:3
                        for ν in μ+1:4
                            rect[threadid() * spacing] +=
                                real(tr(wilsonloop_top_right(U, μ, ν, site, 1, 2))) +
                                real(tr(wilsonloop_top_right(U, μ, ν, site, 2, 1)))
                        end
                    end

                end
            end
        end
    end

    return sum(rect)
end
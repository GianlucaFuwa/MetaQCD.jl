function plaquette_trace_sum(U::Gaugefield{B}) where {B}
    update_halo!(U)

    P = parallelfor_sum(eachindex(U), 0.0, B) do pₙ, site
        for μ in 1i32:3i32
            for ν in (μ+1i32):4i32
                pₙ += real(tr(plaquette(U, μ, ν, site)))
            end
        end
        pₙ
    end

    return distributed_reduce(P, +, U)
end

function rect_trace_sum(U::Gaugefield{B}) where {B}
    is_distributed(U) && @assert(U.topology.halo_width>=2)
    update_halo!(U)

    R = parallelfor_sum(eachindex(U), 0.0, B) do rₙ, site
        for μ in 1:3
            for ν in (μ+1):4
                rₙ += real(tr(rect_1x2(U, μ, ν, site))) + real(tr(rect_2x1(U, μ, ν, site)))
            end
        end
        rₙ
    end

    return distributed_reduce(R, +, U)
end

function plaquette(U, μ, ν, site)
    Nμ = axes(U, μ)
    Nν = axes(U, ν)
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteν⁺ = move(site, ν, 1, Nν)
    plaq = cmatmul_oodd(U[μ, site], U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site])
    return plaq
end

function rect_2x1(U, μ, ν, site)
    Nμ = axes(U, μ)
    Nν = axes(U, ν)
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteμ²⁺ = move(siteμ⁺, μ, 1, Nμ)
    siteμ⁺ν⁺ = move(siteμ⁺, ν, 1, Nν)
    siteν⁺ = move(site, ν, 1, Nν)
    plaq = cmatmul_oo(
        cmatmul_oood(U[μ, site], U[μ, siteμ⁺], U[ν, siteμ²⁺], U[μ, siteμ⁺ν⁺]),
        cmatmul_dd(U[μ, siteν⁺], U[ν, site]),
    )
    return plaq
end

function rect_1x2(U, μ, ν, site)
    Nμ = axes(U, μ)
    Nν = axes(U, ν)
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteμ⁺ν⁺ = move(siteμ⁺, ν, 1, Nν)
    siteν⁺ = move(site, ν, 1, Nν)
    siteν²⁺ = move(siteν⁺, ν, 1, Nν)
    plaq = cmatmul_oo(
        cmatmul_oood(U[μ, site], U[ν, siteμ⁺], U[ν, siteμ⁺ν⁺], U[μ, siteν²⁺]),
        cmatmul_dd(U[ν, siteν⁺], U[ν, site]),
    )
    return plaq
end

function plaquette_2x2(U, μ, ν, site)
    Nμ = axes(U, μ)
    Nν = axes(U, ν)
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteμ²⁺ = move(siteμ⁺, μ, 1, Nμ)
    siteμ²⁺ν⁺ = move(siteμ²⁺, ν, 1, Nν)
    siteν⁺ = move(site, ν, 1, Nν)
    siteν²⁺ = move(siteν⁺, ν, 1, Nν)
    siteμ⁺ν²⁺ = move(siteν²⁺, μ, 1, Nμ)
    plaq = cmatmul_oo(
        cmatmul_oooo(U[μ, site], U[μ, siteμ⁺], U[ν, siteμ²⁺], U[ν, siteμ²⁺ν⁺]),
        cmatmul_dddd(U[μ, siteμ⁺ν²⁺], U[μ, siteν²⁺], U[ν, siteν⁺], U[ν, site]),
    )
    return plaq
end

function plaquette_trace_eachsite(U::Gaugefield{B}) where {B}
    out = zeros(size(U.U))

    parallelfor(eachindex(U), B) do site
        P = 0.0
        for μ in 1:3
            for ν in (μ+1):4
                P += real(tr(plaquette(U, μ, ν, site)))
            end
        end
        out[site] = P
    end

    return out
end


staple(U::Gaugefield{GA}, μ, site) where {GA} = staple(GA(), U, μ, site)

function staple(::WilsonGaugeAction, U, μ, site)
    return staple_plaq(U, μ, site)
end

function staple(::SymanzikTreeGaugeAction, U, μ, site)
    c1 = -1/12
    c1prime = c1
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1)*staple_p + c1prime*staple_r
end

function staple(::SymanzikTadGaugeAction, U, μ, site) # TODO
    # u0sq = sqrt(plaquette_trace_sum(U))
    c1 = -1/12
    c1prime = c1
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1)*staple_p + c1prime*staple_r
end

function staple(::IwasakiGaugeAction, U, μ, site)
    c1 = -0.331
    c1prime = c1
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1)*staple_p + c1prime*staple_r
end

function staple(::DBW2GaugeAction, U, μ, site)
    c1 = -1.409
    c1prime = c1
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1)*staple_p + c1prime*staple_r
end

function staple_plaq(U, μ, site)
    Nμ = size(U)[μ]
    siteμ⁺ = move(site, μ, 1, Nμ)
    staple = @SMatrix zeros(ComplexF64, 3, 3)

    for ν in 1:4
        if ν == μ
            continue
        end

        Nν = size(U)[ν]
        siteν⁺ = move(site, ν, 1, Nν)
        siteν⁻ = move(site, ν, -1, Nν)
        siteμ⁺ν⁻ = move(siteμ⁺, ν, -1, Nν)
        staple += cmatmul_ood(U[ν][site], U[μ][siteν⁺], U[ν][siteμ⁺])
        staple += cmatmul_doo(U[ν][siteν⁻], U[μ][siteν⁻], U[ν][siteμ⁺ν⁻])
    end

    return staple
end

function staple_rect(U, μ, site)
    Nμ = size(U)[μ]
    siteμ⁺  = move(site, μ, 1, Nμ)
    siteμ⁻  = move(site, μ, -1, Nμ)
    siteμ²⁺ = move(site, μ, 2, Nμ)

    staple = @SMatrix zeros(ComplexF64, 3, 3)

    for ν in 1:4
        if ν == μ
            continue
        end

        Nν = size(U)[ν]
        siteν⁺    = move(site, ν, 1, Nν)
        siteν⁻    = move(site, ν, -1, Nν)
        siteμ⁺ν⁺  = move(siteμ⁺, ν, 1, Nν)
        siteμ⁺ν⁻  = move(siteμ⁺, ν, -1, Nν)
        siteμ⁻ν⁺  = move(siteμ⁻, ν, 1, Nν)
        siteμ⁻ν⁻  = move(siteμ⁻, ν, -1, Nν)
        siteν²⁺   = move(site, ν, 2, Nν)
        siteν²⁻   = move(site, ν, -2, Nν)
        siteμ⁺ν²⁻ = move(siteν²⁻, μ, 1, Nμ)
        siteμ²⁺ν⁻ = move(siteν⁻, μ, 2, Nμ)

        # First term, orthogonal to link
        # |- → -|
        # ↑     ↓
        # |-   -|
        # ↑     ↓
        # |-----|
        # ↓     ↑
        # |-   -|
        # ↓     ↑
        # |- → -|
        staple += cmatmul_oo(cmatmul_ooo(U[ν][site], U[ν][siteν⁺], U[μ][siteν²⁺]),
                             cmatmul_dd(U[ν][siteμ⁺ν⁺], U[ν][siteμ⁺]))
        staple += cmatmul_oo(cmatmul_ddo(U[ν][siteν⁻], U[ν][siteν²⁻], U[μ][siteν²⁻]),
                             cmatmul_oo(U[ν][siteμ⁺ν²⁻], U[ν][siteμ⁺ν⁻]))

        # Second term, same direction from site and siteν⁻
        # |- → -|- → -|
        # ↑           ↓
        # |-----|- ← -|
        # ↓           ↑
        # |- → -|- → -|
        staple += cmatmul_od(cmatmul_oood(U[ν][site], U[μ][siteν⁺], U[μ][siteμ⁺ν⁺], U[ν][siteμ²⁺]) +
                              cmatmul_dooo(U[ν][siteν⁻], U[μ][siteν⁻], U[μ][siteμ⁺ν⁻], U[ν][siteμ²⁺ν⁻]),
                             U[μ][siteμ⁺])

        # Third term, same direction from siteμ⁻ and siteμ⁻ν⁻
        # |- → -|- → -|
        # ↑           ↓
        # |- ← -|-----|
        # ↓           ↑
        # |- → -|- → -|
        staple += cmatmul_do(U[μ][siteμ⁻],
                             cmatmul_oood(U[ν][siteμ⁻], U[μ][siteμ⁻ν⁺], U[μ][siteν⁺], U[ν][siteμ⁺]) +
                              cmatmul_dooo(U[ν][siteμ⁻ν⁻], U[μ][siteμ⁻ν⁻], U[μ][siteν⁻], U[ν][siteμ⁺ν⁻]))
    end

    return staple
end

function staple_eachsite!(staples, U)
    @batch per=thread for site in eachindex(U)
        for μ in 1:4
            staples[μ][site] = staple(U, μ, site)
        end
    end

    return nothing
end

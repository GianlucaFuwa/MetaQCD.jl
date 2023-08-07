staple(U::Gaugefield{GA}, μ, site) where {GA} = staple(GA(), U, μ, site)

function staple(::WilsonGaugeAction, U, μ, site)
    return staple_plaq(U, μ, site)
end

function staple(::SymanzikTreeGaugeAction, U, μ, site)
    c1 = -1/12
    c1prime = c1
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1) * staple_p + c1prime * staple_r
end

function staple(::SymanzikTadGaugeAction, U, μ, site)
    # u0sq = sqrt(plaquette_trace_sum(U))
    c1 = -1/12
    # c1prime = c1 / u0sq
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1) * staple_p + c1prime * staple_r
end

function staple(::IwasakiGaugeAction, U, μ, site)
    c1 = -0.331
    c1prime = c1
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1) * staple_p + c1prime * staple_r
end

function staple(::DBW2GaugeAction, U, μ, site)
    c1 = -1.409
    c1prime = c1
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1) * staple_p + c1prime * staple_r
end

function staple_plaq(U, μ, site)
    Nμ = size(U)[μ]
    siteμp = move(site, μ, 1, Nμ)
    staple = @SMatrix zeros(ComplexF64, 3, 3)

    for ν in 1:4
        if ν == μ
            continue
        end

        Nν = size(U)[ν]
        siteνp = move(site, ν, 1, Nν)
        siteνn = move(site, ν, -1, Nν)
        siteμpνn = move(siteμp, ν, -1, Nν)
        staple += cmatmul_ood(U[ν][site], U[μ][siteνp], U[ν][siteμp])
        staple += cmatmul_doo(U[ν][siteνn], U[μ][siteνn], U[ν][siteμpνn])
    end

    return staple
end

function staple_rect(U, μ, site)
    Nμ = size(U)[μ]
    siteμp = move(site, μ, 1, Nμ)
    siteμn = move(site, μ, -1, Nμ)
    site2μp = move(site, μ, 2, Nμ)

    staple = @SMatrix zeros(ComplexF64, 3, 3)

    for ν in 1:4
        if ν == μ
            continue
        end

        Nν = size(U)[ν]
        siteνp = move(site, ν, 1, Nν)
        siteνn = move(site, ν, -1, Nν)
        siteμpνp = move(siteμp, ν, 1, Nν)
        siteμpνn = move(siteμp, ν, -1, Nν)
        siteμnνp = move(siteμn, ν, 1, Nν)
        siteμnνn = move(siteμn, ν, -1, Nν)
        site2νp = move(site, ν, 2, Nν)
        site2νn = move(site, ν, -2, Nν)
        siteμp2νn = move(site2νn, μ, 1, Nμ)
        site2μpνn = move(siteνn, μ, 2, Nμ)

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
        staple += cmatmul_oo(
            cmatmul_ooo(U[ν][site], U[ν][siteνp], U[μ][site2νp]),
            cmatmul_dd(U[ν][siteμpνp], U[ν][siteμp]),
        )
        staple += cmatmul_oo(
            cmatmul_ddo(U[ν][siteνn], U[ν][site2νn], U[μ][site2νn]),
            cmatmul_oo(U[ν][siteμp2νn], U[ν][siteμpνn]),
        )

        # Second term, same direction from site and siteνn
        # |- → -|- → -|
        # ↑           ↓
        # |-----|- ← -|
        # ↓           ↑
        # |- → -|- → -|
        staple += cmatmul_od(
            cmatmul_oood(U[ν][site], U[μ][siteνp], U[μ][siteμpνp], U[ν][site2μp]) +
            cmatmul_dooo(U[ν][siteνn], U[μ][siteνn], U[μ][siteμpνn], U[ν][site2μpνn]),
            U[μ][siteμp],
        )

        # Third term, same direction from siteμn and siteμnνn
        # |- → -|- → -|
        # ↑           ↓
        # |- ← -|-----|
        # ↓           ↑
        # |- → -|- → -|
        staple += cmatmul_do(
            U[μ][siteμn],
            cmatmul_oood(U[ν][siteμn], U[μ][siteμnνp], U[μ][siteνp], U[ν][siteμp]) +
            cmatmul_dooo(U[ν][siteμnνn], U[μ][siteμnνn], U[μ][siteνn], U[ν][siteμpνn]),
        )
    end

    return staple
end

function staple_eachsite!(staples, U)
    @batch for site in eachindex(U)
        for μ in 1:4
            staples[μ][site] = staple(U, μ, site)
        end
    end

    return nothing
end

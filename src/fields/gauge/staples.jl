function staple_eachsite!(staples, U::Gaugefield{T}) where {T}
    NX, NY, NZ, NT = size(U)
    staple = T()

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)
                    for μ in 1:4
                        staples[μ][ix,iy,iz,it] = staple(U, μ, site)
                    end
                end
            end
        end
    end

    return nothing
end

function staple_plaq(U::T, μ, site::SiteCoords) where {T<:Gaugefield}
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
        staple += U[ν][site] * U[μ][siteνp] * U[ν][siteμp]' +
                  U[ν][siteνn]' * U[μ][siteνn] * U[ν][siteμpνn]
    end

    return staple
end

function staple_rect(U::T, μ, site::SiteCoords) where {T<:Gaugefield}
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
        staple += (
            U[ν][site] * U[ν][siteνp] * U[μ][site2νp] * U[ν][siteμpνp]' * U[ν][siteμp]' +
            U[ν][siteνn]' * U[ν][site2νn]' * U[μ][site2νn] * U[ν][siteμp2νn] * U[ν][siteμpνn]
        )

        # Second term, same direction from site and siteνn
        # |- → -|- → -|
        # ↑           ↓
        # |-----|- ← -|
        # ↓           ↑
        # |- → -|- → -|
        staple += (
            U[ν][site] * U[μ][siteνp] * U[μ][siteμpνp] * U[ν][site2μp]' +
            U[ν][siteνn]' * U[μ][siteνn] * U[μ][siteμpνn] * U[ν][site2μpνn]
        ) * U[μ][siteμp]'

        # Third term, same direction from siteμn and siteμnνn
        # |- → -|- → -|
        # ↑           ↓
        # |- ← -|-----|
        # ↓           ↑
        # |- → -|- → -|
        staple += U[μ][siteμn]' * (
            U[ν][siteμn] * U[μ][siteμnνp] * U[μ][siteνp] * U[ν][siteμp]' +
            U[ν][siteμnνn]' * U[μ][siteμnνn] * U[μ][siteνn] * U[ν][siteμpνn]
        )
    end

    return staple
end

function (::WilsonGaugeAction)(U::T, μ, site::SiteCoords) where {T<:Gaugefield}
    return staple_plaq(U, μ, site)
end

function (::SymanzikTreeGaugeAction)(U::T, μ, site::SiteCoords) where {T<:Gaugefield}
    c1 = -1/12
    c1prime = c1
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1) * staple_p + c1prime * staple_r
end

function (::SymanzikTadGaugeAction)(U::T, μ, site::SiteCoords) where {T<:Gaugefield}
    u0sq = sqrt(plaquette_trace_sum(U))
    c1 = -1/12
    c1prime = c1 / u0sq
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1) * staple_p + c1prime * staple_r
end

function (::IwasakiGaugeAction)(U::T, μ, site::SiteCoords) where {T<:Gaugefield}
    c1 = -0.331
    c1prime = c1
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1) * staple_p + c1prime * staple_r
end

function (::DBW2GaugeAction)(U::T, μ, site::SiteCoords) where {T<:Gaugefield}
    c1 = -1.409
    c1prime = c1
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1) * staple_p + c1prime * staple_r
end
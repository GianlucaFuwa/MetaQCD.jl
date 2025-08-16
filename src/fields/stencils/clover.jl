function clover_1x1(U::GaugeLikeField{B,T}, μ, ν, site) where {B,T}
    Nμ = axes(U, μ)
    Nν = axes(U, ν)
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteμ⁻ = move(site, μ, -1, Nμ)
    siteν⁺ = move(site, ν, 1, Nν)
    siteν⁻ = move(site, ν, -1, Nν)
    siteμ⁻ν⁺ = move(siteν⁺, μ, -1, Nμ)
    siteμ⁻ν⁻ = move(siteν⁻, μ, -1, Nμ)
    siteμ⁺ν⁻ = move(siteν⁻, μ, 1, Nμ)

    # @inbounds begin
        # top right
        clover = cmatmul_oodd(U[μ, site], U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site])
        # top left
        clover += cmatmul_oddo(U[ν, site], U[μ, siteμ⁻ν⁺], U[ν, siteμ⁻], U[μ, siteμ⁻])
        # bottom left
        clover += cmatmul_ddoo(U[μ, siteμ⁻], U[ν, siteμ⁻ν⁻], U[μ, siteμ⁻ν⁻], U[ν, siteν⁻])
        # bottom right
        clover += cmatmul_dood(U[ν, siteν⁻], U[μ, siteν⁻], U[ν, siteμ⁺ν⁻], U[μ, site])
    # end

    return clover
end

function clover_2x1(U::GaugeLikeField{B,T}, μ, ν, site) where {B,T}
    Nμ = axes(U, μ)
    Nν = axes(U, ν)
    # 1 hop
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteμ⁻ = move(site, μ, -1, Nμ)
    siteν⁺ = move(site, ν, 1, Nν)
    siteν⁻ = move(site, ν, -1, Nν)
    # 2 hop
    siteμ²⁺ = move(siteμ⁺, μ, 1, Nμ)
    siteμ²⁻ = move(siteμ⁻, μ, -1, Nμ)
    siteμ⁺ν⁺ = move(siteμ⁺, ν, 1, Nν)
    siteμ⁺ν⁻ = move(siteμ⁺, ν, -1, Nν)
    siteμ⁻ν⁺ = move(siteμ⁻, ν, 1, Nν)
    siteμ⁻ν⁻ = move(siteμ⁻, ν, -1, Nν)
    # 3 hop
    siteμ²⁺ν⁻ = move(siteμ²⁺, ν, -1, Nν)
    siteμ²⁻ν⁺ = move(siteμ²⁻, ν, 1, Nν)
    siteμ²⁻ν⁻ = move(siteμ²⁻, ν, -1, Nν)

    @inbounds begin
        # top right
        clover = cmatmul_oo(
            cmatmul_oood(U[μ, site], U[μ, siteμ⁺], U[ν, siteμ²⁺], U[μ, siteμ⁺ν⁺]),
            cmatmul_dd(U[μ, siteν⁺], U[ν, site]),
        )
        # top left
        clover += cmatmul_oo(
            cmatmul_oddd(U[ν, site], U[μ, siteμ⁻ν⁺], U[μ, siteμ²⁻ν⁺], U[ν, siteμ²⁻]),
            cmatmul_oo(U[μ, siteμ²⁻], U[μ, siteμ⁻]),
        )
        # bottom left
        clover += cmatmul_oo(
            cmatmul_dddo(U[μ, siteμ⁻], U[μ, siteμ²⁻], U[ν, siteμ²⁻ν⁻], U[μ, siteμ²⁻ν⁻]),
            cmatmul_oo(U[μ, siteμ⁻ν⁻], U[ν, siteν⁻]),
        )
        # bottom right
        clover += cmatmul_oo(
            cmatmul_dooo(U[ν, siteν⁻], U[μ, siteν⁻], U[μ, siteμ⁺ν⁻], U[ν, siteμ²⁺ν⁻]),
            cmatmul_dd(U[μ, siteμ⁺], U[μ, site]),
        )
    end

    return clover
end

function clover_1x2(U::GaugeLikeField{B,T}, μ, ν, site) where {B,T}
    Nμ = axes(U, μ)
    Nν = axes(U, ν)
    # 1 hop
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteμ⁻ = move(site, μ, -1, Nμ)
    siteν⁺ = move(site, ν, 1, Nν)
    siteν⁻ = move(site, ν, -1, Nν)
    # 2 hop
    siteν²⁺ = move(siteν⁺, ν, 1, Nν)
    siteν²⁻ = move(siteν⁻, ν, -1, Nν)
    siteμ⁺ν⁺ = move(siteν⁺, μ, 1, Nμ)
    siteμ⁻ν⁺ = move(siteν⁺, μ, -1, Nμ)
    siteμ⁺ν⁻ = move(siteν⁻, μ, 1, Nμ)
    siteμ⁻ν⁻ = move(siteν⁻, μ, -1, Nμ)
    # 3 hop
    siteμ⁻ν²⁺ = move(siteν²⁺, μ, -1, Nμ)
    siteμ⁺ν²⁻ = move(siteν²⁻, μ, 1, Nμ)
    siteμ⁻ν²⁻ = move(siteν²⁻, μ, -1, Nμ)

    @inbounds begin
        # top right
        clover = cmatmul_oo(
            cmatmul_oood(U[μ, site], U[ν, siteμ⁺], U[ν, siteμ⁺ν⁺], U[μ, siteν²⁺]),
            cmatmul_dd(U[ν, siteν⁺], U[ν, site]),
        )
        # top left
        clover += cmatmul_oo(
            cmatmul_oodd(U[ν, site], U[ν, siteν⁺], U[μ, siteμ⁻ν²⁺], U[ν, siteμ⁻ν⁺]),
            cmatmul_do(U[ν, siteμ⁻], U[μ, siteμ⁻]),
        )
        # bottom left
        clover += cmatmul_oo(
            cmatmul_dddo(U[μ, siteμ⁻], U[ν, siteμ⁻ν⁻], U[ν, siteμ⁻ν²⁻], U[μ, siteμ⁻ν²⁻]),
            cmatmul_oo(U[ν, siteν²⁻], U[ν, siteν⁻]),
        )
        # bottom right
        clover += cmatmul_oo(
            cmatmul_ddoo(U[ν, siteν⁻], U[ν, siteν²⁻], U[μ, siteν²⁻], U[ν, siteμ⁺ν²⁻]),
            cmatmul_od(U[ν, siteμ⁺ν⁻], U[μ, site]),
        )
    end

    return clover
end

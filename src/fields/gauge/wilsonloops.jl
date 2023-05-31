function wilsonloop(U, μ, ν, site::SiteCoords, Lμ, Lν)
    right = sign(Lμ) == 1
    top = sign(Lν) == 1

    if right && top 
        return wilsonloop_top_right(U, μ, ν, site, Lμ, Lν)
    elseif !right && top
        return wilsonloop_top_left(U, μ, ν, site, Lμ, Lν)
    elseif right && !top
        return wilsonloop_bottom_right(U, μ, ν, site, Lμ, Lν)
    elseif !right && !top 
        return wilsonloop_bottom_left(U, μ, ν, site, Lμ, Lν)
    end

end

function wilsonloop_top_right(U, μ, ν, site::SiteCoords, Lμ, Lν)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    wil = eye3

    for step in 1:Lμ
        wil *= U[μ][site]
        site = move(site, μ, 1, Nμ)
    end

    for step in 1:Lν
        wil *= U[ν][site] 
        site = move(site, ν, 1, Nν)
    end

    for step in 1:Lμ
        site = move(site, μ, -1, Nμ)
        wil *= U[μ][site]'
    end

    for step in 1:Lν
        site = move(site, ν, -1, Nν)
        wil *= U[ν][site]'
    end

    return wil
end

function wilsonloop_bottom_left(U, μ, ν, site::SiteCoords, Lμ, Lν)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    wil = eye3

    for step in 1:Lμ
        site = move(site, μ, -1, Nμ)
        wil *= U[μ][site]'
    end

    for step in 1:Lν
        site = move(site, ν, -1, Nν)
        wil *= U[ν][site]' 
    end

    for step in 1:Lμ
        wil *= U[μ][site]
        site = move(site, μ, 1, Nμ)
    end

    for step in 1:Lν
        wil *= U[ν][site]
        site = move(site, ν, 1, Nν)
    end

    return wil
end

function wilsonloop_top_left(U, μ, ν, site::SiteCoords, Lμ, Lν)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    wil = eye3

    for step in 1:Lν
        wil *= U[ν][site]
        site = move(site, ν, 1, Nν)
    end

    for step in 1:Lμ
        site = move(site, μ, -1, Nμ)
        wil *= U[μ][site]'
    end

    for step in 1:Lν
        site = move(site, ν, -1, Nν)
        wil *= U[ν][site]'
    end

    for step in 1:Lμ
        wil *= U[μ][site]
        site = move(site, μ, 1, Nμ)
    end

    return wil
end

function wilsonloop_bottom_right(U, μ, ν, site::SiteCoords, Lμ, Lν)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    wil = eye3

    for step in 1:Lν
        site = move(site, ν, -1, Nν)
        wil *= U[ν][site]' 
    end

    for step in 1:Lμ
        wil *= U[μ][site]
        site = move(site, μ, 1, Nμ)
    end

    for step in 1:Lν
        wil *= U[ν][site]
        site = move(site, ν, 1, Nν)
    end

    for step in 1:Lμ
        site = move(site, μ, -1, Nμ)
        wil *= U[μ][site]'
    end

    return wil
end

function wilsonloop(U::T, Lμ, Lν) where {T<:Gaugefield}
    NX, NY, NZ, NT = size(U)
    wil = 0.0

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)
                    for μ in 1:3
                        for ν in μ+1:4
                            wil += real(
                                tr(wilsonloop(U, μ, ν, site, Lμ, Lν))
                            )
                        end
                    end
                end
            end
        end
    end

    return wil
end
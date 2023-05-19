function staple_eachsite!(staples::TemporaryField, U::Gaugefield)
    kind_of_gaction = U.kind_of_gaction
    
    if kind_of_gaction == "wilson"
        staple_eachsite_plaq!(staples, U)
    elseif kind_of_gaction == "symanzik" 
        staple_eachsite_symanzik!(staples, U)
    elseif kind_of_gaction == "iwasaki" 
        staple_eachsite_iwasaki!(staples, U)
    elseif kind_of_gaction == "dbw2" 
        staple_eachsite_dbw2!(staples, U)
    else 
        error("type_of_gaction $(U.kind_of_gaction) is not supported!")
    end
    
    return nothing
end

function staple_eachsite_plaq!(staples::TemporaryField, U::Gaugefield)
    NX, NY, NZ, NT = size(U)

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)
                    for μ in 1:4
                        staples[μ][ix,iy,iz,it] = staple_plaq(U, μ, site)
                    end
                end
            end
        end
    end

    return nothing
end

function staple_eachsite_rect!(staples::TemporaryField, U::Gaugefield)
    NX, NY, NZ, NT = size(U)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix,iy,iz,it)
                    for μ in 1:4
                        staples[μ][ix,iy,iz,it] = staple_rect(U, μ, site)
                    end
                end
            end
        end
    end

    return nothing
end

function staple_eachsite_symanzik!(staples::TemporaryField, U::Gaugefield)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)
                    for μ in 1:4
                        staples[μ][ix,iy,iz,it] = staple_symanzik(U, μ, site)
                    end
                end
            end
        end
    end

    return nothing
end

function staple_eachsite_iwasaki!(staples::TemporaryField, U::Gaugefield)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)
                    for μ in 1:4
                        staples[μ][ix,iy,iz,it] = staple_iwasaki(U, μ, site)
                    end
                end
            end
        end
    end

    return nothing
end

function staple_eachsite_dbw2!(staples::TemporaryField, U::Gaugefield)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)
                    for μ in 1:4
                        staples[μ][ix,iy,iz,it] = staple_dbw2(U, μ, site)
                    end
                end
            end
        end
    end

    return nothing
end

function staple_eachsite_improved_topcharge!(staples::TemporaryField, U::Gaugefield)
    NX, NY, NZ, NT = size(U)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)
                    for μ in 1:4
                        staples[μ][ix,iy,iz,it] = staple_improved_topcharge(U, μ, site)
                    end
                end
            end
        end
    end

    return nothing
end

function staple(U::Gaugefield, μ, site::SiteCoords)
    
    if U.kind_of_gaction == "wilson"
        staple = staple_plaq(U, μ, site)
    elseif U.kind_of_gaction == "symanzik"
        staple = staple_symanzik(U, μ, site)
    elseif U.kind_of_gaction == "iwasaki"
        staple = staple_iwasaki(U, μ, site)
    elseif U.kind_of_gaction == "dbw2"
        staple = staple_dbw2(U, μ, site)
    else 
        error("type_of_gaction $(U.kind_of_gaction) is not supported!")
    end

    return staple
end

function staple_plaq(U::Gaugefield, μ, site::SiteCoords)
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

function staple_rect(U::Gaugefield, μ, site::SiteCoords)
    Nμ = size(U)[μ]
    siteμp = move(site, μ, 1, Nμ)
    site2μp = move(site, μ, 2, Nμ)
    staple = @SMatrix zeros(ComplexF64, 3, 3)

    for ν in 1:4
        if ν == μ
            continue
        end

        Nν = size(U)[ν]
        siteνp = move(site, ν, 1, Nν)
        siteνn = move(site, ν, -1, Nν)
        siteμpνn = move(siteμp, ν, -1, Nν)
        siteμpνp = move(siteμp, ν, 1, Nν)
        site2νn = move(site, ν, -1, Nν)
        siteμp2νn = move(site2νn, μ, 1, Nμ)
        site2μpνn = move(siteνn, μ, 2, Nμ)
        staple += U[ν][site] * U[ν][siteνp] * U[μ][siteνp] * U[ν][siteμpνp]' * U[ν][siteμp]' +
            U[ν][siteνn]' * U[ν][site2νn]' * U[μ][site2νn] * U[ν][siteμp2νn] * U[ν][siteμpνn]
        staple += U[ν][site] * U[μ][siteνp] * U[μ][siteμpνp] * U[ν][site2μp]' * U[μ][siteμp]' +
            U[ν][siteνn]' * U[μ][siteνn] * U[μ][siteμpνn] * U[ν][site2μpνn] * U[ν][siteμp]'
    end

    return staple
end

function staple_symanzik(U::Gaugefield, μ, site::SiteCoords)
    u0sq = sqrt(plaquette_trace_sum(U))
    c1 = -1/12
    c1prime = c1 / u0sq
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1) * staple_p + c1prime * staple_r
end

function staple_iwasaki(U::Gaugefield, μ, site::SiteCoords)
    c1 = -0.331
    c1prime = c1
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1) * staple_p + c1prime * staple_r
end

function staple_dbw2(U::Gaugefield, μ, site::SiteCoords)
    c1 = -1.409
    c1prime = c1
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return (1 - 8c1) * staple_p + c1prime * staple_r
end

function staple_improved_topcharge(U::Gaugefield, μ, site::SiteCoords)
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return 5/3*staple_p - 1/6*staple_r
end
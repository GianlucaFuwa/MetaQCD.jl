function staple_eachsite!(staples::T1, U::T2, kind_of_CV=nothing) where {T1<:Temporary_field, T2<:Gaugefield}
    if kind_of_CV !== nothing
        if kind_of_CV == "Plaquette"
            staple_eachsite_plaq!(staples, U)
        elseif kind_of_CV == "Clover" 
            staple_eachsite_plaq!(staples, U)
        elseif kind_of_CV == "Improved" 
            staple_eachsite_improved_topcharge!(staples, U)
        else 
            error("kind_of_CV $(kind_of_CV) is not supported!")
        end
    else
        kind_of_gaction = get_gaction(U)
        if kind_of_gaction == "Wilson"
            staple_eachsite_plaq!(staples, U)
        elseif kind_of_gaction == "Symanzik" 
            staple_eachsite_improved!(staples, U)
        elseif kind_of_gaction == "Iwasaki" 
            staple_eachsite_improved!(staples, U)
        elseif kind_of_gaction == "DBW2" 
            staple_eachsite_improved!(staples, U)
        else 
            error("type_of_gaction $(U.kind_of_gaction) is not supported!")
        end
    end
    return nothing
end

function staple_eachsite_plaq!(staples::T1, U::T2) where {T1<:Temporary_field, T2<:Gaugefield}
    NX, NY, NZ, NT = size(U)
    @batch for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    site = Site_coords(ix,iy,iz,it)
                    for μ = 1:4
                        #staples[μ][ix,iy,iz,it] = staple_plaq(U, μ, ix, iy, iz, it)
                        staples[μ][ix,iy,iz,it] = staple_plaq(U, μ, site)
                    end
                end
            end
        end
    end
    return nothing
end

function staple_eachsite_rect!(staples::T1, U::T2) where {T1<:Temporary_field, T2<:Gaugefield}
    NX, NY, NZ, NT = size(U)
    @batch for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    site = Site_coords(ix,iy,iz,it)
                    for μ = 1:4
                        staples[μ][ix,iy,iz,it] = staple_rect(U, μ, site)
                    end
                end
            end
        end
    end
    return nothing
end

function staple_eachsite_improved!(staples::T1, U::T2) where {T1<:Temporary_field, T2<:Gaugefield}
    NX,NY,NZ,NT = size(U)
    @batch for it=1:NT
        for iz=1:NZ
            for iy=1:NY
                for ix=1:NX
                    site = Site_coords(ix,iy,iz,it)
                    for μ=1:4
                        staples[μ][ix,iy,iz,it] = staple_improved(U, μ, site)
                    end
                end
            end
        end
    end
    return nothing
end

function staple_eachsite_improved_topcharge!(staples::T1, U::T2) where {T1<:Temporary_field, T2<:Gaugefield}
    NX,NY,NZ,NT = size(U)
    @batch for it=1:NT
        for iz=1:NZ
            for iy=1:NY
                for ix=1:NX
                    site = Site_coords(ix,iy,iz,it)
                    for μ=1:4
                        staples[μ][ix,iy,iz,it] = staple_improved_topcharge(U, μ, site)
                    end
                end
            end
        end
    end
    return nothing
end

function staple(U::T, μ, site::Site_coords) where {T<:Gaugefield}
    kind_of_gaction = get_gaction(U)

    if kind_of_gaction == "Wilson"
        staple = staple_plaq(U, μ, site)
    elseif any(x -> x == kind_of_gaction,["Symanzik", "Iwasaki", "DBW2"])
        staple = staple_improved(U, μ, site)
    else 
        error("type_of_gaction $(U.kind_of_gaction) is not supported!")
    end

    return staple
end

function staple_plaq(U::T, μ, site::Site_coords) where {T<:Gaugefield}
    Nμ = size(U)[μ]
    siteμp = move(site, μ, 1, Nμ)
    staple = SMatrix{3,3,ComplexF64,9}(zeros(ComplexF64,3,3))
    for ν = 1:4
        if ν == μ
            continue
        end
        Nν = size(U)[ν]
        siteνp = move(site, ν, 1, Nν)
        siteνn = move(site, ν, -1, Nν)
        siteμpνn = move(siteμp, ν, -1, Nν)
        staple += U[ν][site]  * U[μ][siteνp] * U[ν][siteμp]' +
                  U[ν][siteνn]' * U[μ][siteνn] * U[ν][siteμpνn]
    end
    return staple
end

function staple_rect(U::T, μ, site::Site_coords) where {T<:Gaugefield}
    Nμ = size(U)[μ]
    siteμp = move(site, μ, 1, Nμ)
    site2μp = move(site, μ, 2, Nμ)
    staple = SMatrix{3,3,ComplexF64,9}(zeros(ComplexF64,3,3))
    for ν=1:4
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
        staple += U[ν][site]  * U[ν][siteνp] * U[μ][siteνp] * U[ν][siteμpνp]' * U[ν][siteμp]' +
                  U[ν][siteνn]' * U[ν][site2νn]' * U[μ][site2νn] * U[ν][siteμp2νn] * U[ν][siteμpνn]
        staple += U[ν][site]  * U[μ][siteνp] * U[μ][siteμpνp] * U[ν][site2μp]' * U[μ][siteμp]' +
                  U[ν][siteνn]' * U[μ][siteνn] * U[μ][siteμpνn] * U[ν][site2μpνn] * U[ν][siteμp]'
    end
    return staple
end

function staple_improved(U::T, μ, site::Site_coords) where {T<:Gaugefield}
    kind_of_gaction = get_gaction(U)
    if kind_of_gaction == "Symanzik"
        u0sq = sqrt(plaquette_tracedsum(U))
        c1 = -1/12
        c1prime = c1/u0sq
        staple_p = staple_plaq(U, μ, site)
        staple_r = staple_rect(U, μ, site)
    elseif kind_of_gaction == "Iwasaki"
        c1 = -0.331
        c1prime = c1
        staple_p = staple_plaq(U, μ, site)
        staple_r = staple_rect(U, μ, site)
    elseif kind_of_gaction == "DBW2"
        c1 = -1.409
        c1prime = c1
        staple_p = staple_plaq(U, μ, site)
        staple_r = staple_rect(U, μ, site)
    end
    return (1-8c1)staple_p + c1prime*staple_r
end

function staple_improved_topcharge(U::T, μ, site::Site_coords) where {T<:Gaugefield}
    staple_p = staple_plaq(U, μ, site)
    staple_r = staple_rect(U, μ, site)
    return 5/3*staple_p - 1/6*staple_r
end
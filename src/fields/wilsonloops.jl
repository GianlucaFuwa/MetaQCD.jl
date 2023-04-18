function wilsonloop(g::T,μ,ν,site::Site_coords;Lμ=1,Lν=1) where {T<:Gaugefield}
    right = sign(Lμ)==1
    top = sign(Lν)==1

    if right && top 
        return wilsonloop_top_right(g,μ,ν,site,Lμ,Lν)
    elseif !right && top
        return wilsonloop_top_left(g,μ,ν,site,Lμ,Lν)
    elseif right && !top
        return wilsonloop_bottom_right(g,μ,ν,site,Lμ,Lν)
    elseif !right && !top 
        return wilsonloop_bottom_left(g,μ,ν,site,Lμ,Lν)
    end
end

function wilsonloop_top_right(g::T,μ,ν,site::Site_coords,Lμ,Lν) where {T<:Gaugefield}
    Nμ = size(g)[μ]
    Nν = size(g)[ν]
    wil = SMatrix{3,3,ComplexF64}(I)
    for step = 1:Lμ
        wil *= g[μ][site]
        site = move(site,μ,1,Nμ)
    end
    for step = 1:Lν
        wil *= g[ν][site] 
        site = move(site,ν,1,Nν)
    end
    for step = 1:Lμ
        site = move(site,μ,-1,Nμ)
        wil *= g[μ][site]'
    end
    for step = 1:Lν
        site = move(site,ν,-1,Nν)
        wil *= g[ν][site]'
    end
    return wil
end

function wilsonloop_bottom_left(g::T,μ,ν,site::Site_coords,Lμ,Lν) where {T<:Gaugefield}
    Nμ = size(g)[μ]
    Nν = size(g)[ν]
    wil = SMatrix{3,3,ComplexF64}(I)
    for step = 1:Lμ
        site = move(site,μ,-1,Nμ)
        wil *= g[μ][site]'
    end
    for step = 1:Lν
        site = move(site,ν,-1,Nν)
        wil *= g[ν][site]' 
    end
    for step = 1:Lμ
        wil *= g[μ][site]
        site = move(site,μ,1,Nμ)
    end
    for step = 1:Lν
        wil *= g[ν][site]
        site = move(site,ν,1,Nν)
    end
    return wil
end

function wilsonloop_top_left(g::T,μ,ν,site::Site_coords,Lμ,Lν) where {T<:Gaugefield}
    Nμ = size(g)[μ]
    Nν = size(g)[ν]
    wil = SMatrix{3,3,ComplexF64}(I)
    for step = 1:Lν
        wil *= g[ν][site]
        site = move(site,ν,1,Nν)
    end
    for step = 1:Lμ
        site = move(site,μ,-1,Nμ)
        wil *= g[μ][site]'
    end
    for step = 1:Lν
        site = move(site,ν,-1,Nν)
        wil *= g[ν][site]'
    end
    for step = 1:Lμ
        wil *= g[μ][site]
        site = move(site,μ,1,Nμ)
    end
    return wil
end

function wilsonloop_bottom_right(g::T,μ,ν,site::Site_coords,Lμ,Lν) where {T<:Gaugefield}
    Nμ = size(g)[μ]
    Nν = size(g)[ν]
    wil = SMatrix{3,3,ComplexF64}(I)
    for step = 1:Lν
        site = move(site,ν,-1,Nν)
        wil *= g[ν][site]' 
    end
    for step = 1:Lμ
        wil *= g[μ][site]
        site = move(site,μ,1,Nμ)
    end
    for step = 1:Lν
        wil *= g[ν][site]
        site = move(site,ν,1,Nν)
    end
    for step = 1:Lμ
        site = move(site,μ,-1,Nμ)
        wil *= g[μ][site]'
    end
    return wil
end

function wilsonloop(g::T;Lμ,Lν) where {T<:Gaugefield}
    NX,NY,NZ,NT = size(g)
    wil = 0.0
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    site = Site_coords(ix,iy,iz,it)
                    for μ = 1:3
                        for ν = μ+1:4
                        wil += real(tr( wilsonloop(g,μ,ν,site,Lμ=Lμ,Lν=Lν) ))
                        end
                    end
                end
            end
        end
    end
    return wil
end





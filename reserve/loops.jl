module Loops
    
    struct Loops_set

        function clover_rect(g::T,μ,ν,site::Site_coords) where {T<:Gaugefield}
            Nμ = size(g)[μ]
            Nν = size(g)[ν]
            siteμp = move(site,μ,1,Nμ); siteμn = move(site,μ,-1,Nμ)
            siteνp = move(site,ν,1,Nν); siteνn = move(site,ν,-1,Nν)
            site2μp = move(site,μ,2,Nμ); site2μn = move(site,μ,-2,Nμ)
            site2νp = move(site,ν,2,Nν); site2νn = move(site,ν,-2,Nν)
            siteμpνp = move(move(site,μ,1,Nμ),ν,1,Nν); siteμnνn = move(move(site,μ,-1,Nμ),ν,-1,Nν)
            siteνpμn = move(move(site,μ,-1,Nμ),ν,1,Nν); siteνnμp = move(move(site,μ,1,Nμ),ν,-1,Nν)
            site2μpνn = move(move(site,μ,2,Nμ),ν,-1,Nν); siteμp2νn = move(move(site,μ,1,Nμ),ν,-2,Nν)
            siteνp2μn = move(move(site,μ,-2,Nμ),ν,1,Nν); site2νpμn = move(move(site,μ,-1,Nμ),ν,2,Nν)
            site2νnμn = move(move(site,μ,-2,Nμ),ν,-1,Nν); siteνn2μn = move(move(site,μ,-1,Nμ),ν,-2,Nν);
    
            topright1x2    = g[μ][site] * g[ν][siteμp] * g[ν][siteμpνp]' * g[μ][site2νp]' * g[ν][siteνp]' * g[ν][site]'
    
            topleft1x2     = g[ν][site] * g[ν][siteνp] * g[μ][site2νpμn]' * g[ν][siteνpμn]' * g[ν][siteμn]' * g[μ][siteμn]
    
            bottomleft1x2  = g[μ][siteμn] * g[ν][siteμnνn]' * g[ν][site2νnμn]' * g[μ][site2νnμn] * g[ν][site2νn] * g[μ][siteνn]
    
            bottomright1x2 = g[ν][siteνn]' * g[ν][siteμnνn]' * g[ν][site2νnμn]' * g[μ][site2νnμn] * g[ν][site2νn] * g[μ][siteνn]
    
            topright2x1    = g[μ][site] * g[μ][siteμp] * g[ν][site2μp] * g[μ][siteμpνp]' * g[μ][siteμp]' * g[ν][site]' 
    
            topleft2x1     = g[ν][site] * g[μ][siteνpμn]' * g[μ][siteνp2μn]' * g[ν][site2μn]' * g[μ][site2μn] * g[μ][siteμn]
    
            bottomleft2x1  = g[μ][siteμn]' * g[μ][site2μn]' * g[ν][siteνn2μn]' * g[μ][siteνn2μn] * g[μ][siteμnνn] * g[ν][siteνn]
    
            bottomright2x1 = g[ν][siteνn]' * g[μ][siteνn] * g[μ][siteνnμp] * g[ν][site2μpνn] * g[μ][siteμp]' * g[μ][site]'
            
            return topright1x2 + topleft1x2 + bottomleft1x2 + bottomright1x2 + topright2x1 + topleft2x1 + bottomleft2x1 + bottomright2x1
        end

        function clover(g::T,μ,ν,site::Site_coords) where {T<:Gaugefield}
            Nμ = size(g)[μ]
            Nν = size(g)[ν]
            topright    = g[μ][site] * g[ν][move(site,μ,1,Nμ)] * g[μ][move(site,ν,1,Nν)]' * g[ν][site]'
            topleft     = g[ν][site] * g[μ][move(move(site,ν,1,Nν),μ,-1,Nμ)]' * g[ν][move(site,μ,-1,Nμ)]' * g[μ][move(site,μ,-1,Nμ)] 
            bottomleft  = g[μ][move(site,μ,-1,Nμ)]' * g[ν][move(move(site,ν,-1,Nν),μ,-1,Nμ)]' * g[μ][move(move(site,ν,-1,Nν),μ,-1,Nμ)] * g[ν][move(site,ν,-1,Nν)]
            bottomright = g[ν][move(site,ν,-1,Nν)]' * g[μ][move(site,ν,-1,Nν)] * g[ν][move(move(site,ν,-1,Nν),μ,1,Nμ)] * g[μ][site]'
            return topright + topleft + bottomleft + bottomright
        end

        function plaquette_trsum(g::T) where {T<:Gaugefield}
            space = 8
            plaq = zeros(ComplexF64,nthreads()*space)
            NX,NY,NZ,NT = size(g)
            @batch for it=1:NT
                for iz=1:NZ
                    for iy=1:NY; iy_plu = mod1(iy+1,NY); iz_plu = mod1(iz+1,NZ); it_plu = mod1(it+1,NT);
                        for ix=1:NX; ix_plu = mod1(ix+1,NX);
                        plaq[threadid()*space]+=tr(g[1][ix,iy,iz,it]*g[2][ix_plu,iy,iz,it]*g[1][ix,iy_plu,iz,it]'*g[2][ix,iy,iz,it]') +
                                                tr(g[1][ix,iy,iz,it]*g[3][ix_plu,iy,iz,it]*g[1][ix,iy,iz_plu,it]'*g[3][ix,iy,iz,it]') +
                                                tr(g[2][ix,iy,iz,it]*g[3][ix,iy_plu,iz,it]*g[2][ix,iy,iz_plu,it]'*g[3][ix,iy,iz,it]') +
                                                tr(g[1][ix,iy,iz,it]*g[4][ix_plu,iy,iz,it]*g[1][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]') +	
                                                tr(g[2][ix,iy,iz,it]*g[4][ix,iy_plu,iz,it]*g[2][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]') +	
                                                tr(g[3][ix,iy,iz,it]*g[4][ix,iy,iz_plu,it]*g[3][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]') 
                        end
                    end
                end
            end
            return real(sum(plaq))
        end
    

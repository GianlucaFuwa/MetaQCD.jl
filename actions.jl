module GaugeActions
    using LinearAlgebra
    using Polyester
    #using Base.Threads

    import ..Gaugefields: Gaugefield

    function calc_GaugeAction(g::Vector{T},β;type_of_action="plaq") where {T<:Gaugefield}
        if type_of_action == "plaq"
            Sg = Sg_wils_plaq(g)
            Sg *= β
        else
            error("type_of_action $type_of_action is not supported!")
        end
        return Sg
    end

    function Sg_wils_plaq(g::Vector{T}) where {T<:Gaugefield}
		space = 8
		Sg = zeros(Float64,nthreads()*space)
		NX = g[1].NX
		NY = g[1].NY
		NZ = g[1].NZ
		NT = g[1].NT
		@batch per=thread for it=1:NT
		for iz=1:NZ
		for iy=1:NY; iy_plu = mod1(iy+1,NY); iz_plu = mod1(iz+1,NZ); it_plu = mod1(it+1,NT);
		for ix=1:NX; ix_plu = mod1(ix+1,NX);
			Sg[threadid()*space] += 3.0 - real(tr(g[1][ix,iy,iz,it]*g[2][ix_plu,iy,iz,it]*g[1][ix,iy_plu,iz,it]'*g[2][ix,iy,iz,it]')) +
				3.0 - real(tr(g[2][ix,iy,iz,it]*g[3][ix,iy_plu,iz,it]*g[2][ix,iy,iz_plu,it]'*g[3][ix,iy,iz,it]')) +
				3.0 - real(tr(g[3][ix,iy,iz,it]*g[1][ix,iy,iz_plu,it]*g[3][ix_plu,iy,iz,it]'*g[1][ix,iy,iz,it]')) +
				3.0 - real(tr(g[1][ix,iy,iz,it]*g[4][ix_plu,iy,iz,it]*g[1][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]')) +
				3.0 - real(tr(g[2][ix,iy,iz,it]*g[4][ix,iy_plu,iz,it]*g[2][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]')) +
				3.0 - real(tr(g[3][ix,iy,iz,it]*g[4][ix,iy,iz_plu,it]*g[3][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]'))
		end
		end
		end
		end
		return sum(Sg)/3/g[1].NV
	end

end
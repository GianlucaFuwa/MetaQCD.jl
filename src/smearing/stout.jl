module Stout_smearing
    using StaticArrays
    using LinearAlgebra
    using Polyester
    #using Base.Threads

    import ..Utils:exp_iQ
    import ..Gaugefields:Gaugefield,substitute_U!

    function stout_smear!(g::Gaugefield,gcopy11::Gaugefield,gcopy12::Gaugefield,ρ_stout::Float64)
		NX = g.NX
        NY = g.NY
        NZ = g.NZ
        NT = g.NT

		@batch per=thread for it=1:NT
		for iz=1:NZ
		for iy=1:NY; iy_min = mod1(iy-1,NY); iy_plu = mod1(iy+1,NY);iz_min = mod1(iz-1,NZ); iz_plu = mod1(iz+1,NZ);it_min = mod1(it-1,NT); it_plu = mod1(it+1,NT)
		for ix=1:NX; ix_min = mod1(ix-1,NX); ix_plu = mod1(ix+1,NX);
			tmp=gcopy2[2][ix,iy    ,iz,    it]*gcopy2[1][ix,iy_plu,iz    ,it]*gcopy2[2][ix_plu,iy    ,iz    ,it    ] +
				gcopy2[2][ix,iy_min,iz,    it]*gcopy2[1][ix,iy_min,iz    ,it]*gcopy2[2][ix_plu,iy_min,iz    ,it    ] +
				gcopy2[3][ix,iy    ,iz    ,it]*gcopy2[1][ix,iy    ,iz_plu,it]*gcopy2[3][ix_plu,iy    ,iz    ,it    ] +
				gcopy2[3][ix,iy    ,iz_min,it]*gcopy2[1][ix,iy    ,iz_min,it]*gcopy2[3][ix_plu,iy    ,iz_min,it    ] +
				gcopy2[4][ix,iy    ,iz    ,it]*gcopy2[1][ix,iy    ,iz,it_plu]*gcopy2[4][ix_plu,iy    ,iz    ,it    ] +
				gcopy2[4][ix,iy    ,iz,it_min]*gcopy2[1][ix,iy    ,iz,it_min]*gcopy2[4][ix_plu,iy    ,iz    ,it_min]
			Ω = tmp*ρ_stout*gcopy2[1][ix,iy,iz,it]'
			Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
			gcopy1[1][ix,iy,iz,it] = exp_iQ(Q)*gcopy1[1][ix,iy,iz,it]

			tmp=gcopy2[1][ix,iy    ,iz,    it]*gcopy2[2][ix_plu,iy,iz    ,it    ]*gcopy2[1][ix,iy_plu,iz    ,it    ] +
				gcopy2[1][ix_min,iy,iz,it    ]*gcopy2[2][ix_min,iy,iz    ,it    ]*gcopy2[1][ix_min,iy_plu,iz,it    ] +
				gcopy2[3][ix,iy    ,iz    ,it]*gcopy2[2][ix,iy    ,iz_plu,it    ]*gcopy2[3][ix,iy_plu,iz    ,it    ] +
				gcopy2[3][ix,iy    ,iz_min,it]*gcopy2[2][ix,iy    ,iz_min,it    ]*gcopy2[3][ix,iy_plu,iz_min,it    ] +
				gcopy2[4][ix,iy    ,iz    ,it]*gcopy2[2][ix,iy    ,iz    ,it_plu]*gcopy2[4][ix,iy_plu,iz    ,it    ] +
				gcopy2[4][ix,iy    ,iz,it_min]*gcopy2[2][ix,iy    ,iz    ,it_min]*gcopy2[4][ix,iy_plu,iz    ,it_min]
			Ω = tmp*ρ_stout*gcopy2[2][ix,iy,iz,it]'
			Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
			gcopy1[2][ix,iy,iz,it] = exp_iQ(Q)*gcopy1[2][ix,iy,iz,it]

			tmp=gcopy2[1][ix,iy    ,iz,    it]*gcopy2[3][ix_plu,iy,iz    ,it    ]*gcopy2[1][ix,iy    ,iz_plu,it    ] +
				gcopy2[1][ix_min,iy,iz,    it]*gcopy2[3][ix_min,iy,iz    ,it    ]*gcopy2[1][ix_min,iy,iz_plu,it    ] +
				gcopy2[2][ix,iy    ,iz    ,it]*gcopy2[3][ix,iy_plu,iz    ,it    ]*gcopy2[2][ix,iy    ,iz_plu,it    ] +
				gcopy2[2][ix,iy_min,iz    ,it]*gcopy2[3][ix,iy_min,iz    ,it    ]*gcopy2[2][ix,iy_min,iz_plu,it    ] +
				gcopy2[4][ix,iy    ,iz    ,it]*gcopy2[3][ix,iy    ,iz    ,it_plu]*gcopy2[4][ix,iy    ,iz_plu,it    ] +
				gcopy2[4][ix,iy    ,iz,it_min]*gcopy2[3][ix,iy    ,iz    ,it_min]*gcopy2[4][ix,iy    ,iz_plu,it_min]
			Ω = tmp*ρ_stout*gcopy2[3][ix,iy,iz,it]'
			Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
			gcopy1[3][ix,iy,iz,it] = exp_iQ(Q)*gcopy1[3][ix,iy,iz,it]

			tmp=gcopy2[1][ix,iy    ,iz,    it]*gcopy2[4][ix_plu,iy,iz    ,it    ]*gcopy2[1][ix,iy    ,iz    ,it_plu] +
				gcopy2[1][ix_min,iy,iz,    it]*gcopy2[4][ix_min,iy,iz    ,it    ]*gcopy2[1][ix_min,iy_min,iz,it_plu] +
				gcopy2[2][ix,iy    ,iz    ,it]*gcopy2[4][ix,iy_plu,iz    ,it    ]*gcopy2[2][ix,iy    ,iz    ,it_plu] +
				gcopy2[2][ix,iy_min,iz    ,it]*gcopy2[4][ix,iy_min,iz    ,it    ]*gcopy2[2][ix,iy_min,iz    ,it_plu] +
				gcopy2[3][ix,iy    ,iz    ,it]*gcopy2[4][ix,iy    ,iz_plu,it    ]*gcopy2[3][ix,iy    ,iz    ,it_plu] +
				gcopy2[3][ix,iy    ,iz_min,it]*gcopy2[4][ix,iy    ,iz_min,it    ]*gcopy2[3][ix,iy    ,iz_min,it_plu]
			Ω = tmp*ρ_stout*gcopy2[4][ix,iy,iz,it]'
			Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
			gcopy1[4][ix,iy,iz,it] = exp_iQ(Q)*gcopy1[4][ix,iy,iz,it]
		end
		end
		end
		end
		substitute_U!(gfieldcopy2,gfieldcopy1)
		return nothing
	end

end
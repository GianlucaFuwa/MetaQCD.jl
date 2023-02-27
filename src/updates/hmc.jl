module HMC
    using Random 
    using StaticArrays
    using LinearAlgebra
    using Polyester
    #using Base.Threads

    import ..System_parameters:Params
    import ..Utils:exp_iQ
    import ..Verbose_print:Verbose_,println_verbose
    import ..Gaugefields:Gaugefield,calc_Sgwils,recalc_CV!
    import ..Liefields:Liefield,gaussianP!,trP2

    function HMC!(gfield::Gaugefield,pfield::Liefield,ϵ_hmc::Float64,hmc_steps::Int64,metro_test::Bool,rng::Xoshiro,verbose::Verbose)
        Uold = deepcopy(gfield.U)

        gaussianP!(pfield,rng)
        Sg_old = gfield.Sg
        trP2_old = trP2(pfield)

        updateP!(gfield,pfield,ϵ_hmc,0.5)
        for step = 1:hmc_steps
            updateU!(gfield,pfield,ϵ_hmc,1.0)
            updateP!(gfield,pfield,ϵ_hmc,1.0)
        end
        updateP!(gfield,pfield,ϵ_hmc,0.5)

        Sg_new = calc_Sgwils(gfield)
        trP2_new = trP2(pfield)

        ΔH = trP2_old/2 - trP2_new/2  + Sg_old - Sg_new
        if metro_test
            accept = rand(rng)≤exp( ΔH )
            if accept
            else
                gfield.U = Uold
            end
        else
            println_verbose(verbose,"ΔH = ",ΔH)
        end

        return accept
    end

    function HMCmeta!(gfield::Gaugefield,pfield::Liefield,ϵ_hmc::Float64,hmc_steps::Int64,metro_test::Bool,rng::Xoshiro,verbose::Verbose)
        Uold = deepcopy(gfield.U)

        gaussianP!(pfield,rng)
        Sg_old = gfield.Sg
        trP2_old = trP2(pfield)

        updateP!(gfield,pfield,ϵ_hmc,0.5)
        for step = 1:hmc_steps
            updateU!(gfield,pfield,ϵ_hmc,1.0)
            updateP!(gfield,pfield,ϵ_hmc,1.0)
        end
        updateP!(gfield,pfield,ϵ_hmc,0.5)

        Sg_new = calc_Sgwils(gfield)
        trP2_new = trP2(pfield)

        accept = metro_test ? rand(rng)≤exp( trP2_old/2 - trP2_new/2  + Sg_old - Sg_new ) :  true 
        if accept
            recalc_CV!(gfield)
        else
            gfield.U = Uold
        end
        return accept
    end

    function updateU!(gfield::Gaugefield,pfield::Liefield,ϵ_hmc::Float64,fac::Float64)
        NX = gfield.NX
        NY = gfield.NY
        NZ = gfield.NZ
        NT = gfield.NT
        ϵ = ϵ_hmc*fac
        @batch per=thread for it=1:NT
        for iz=1:NZ
        for iy=1:NY
        for ix=1:NX
            for μ=1:4
                gfield[μ][ix,iy,iz,it] = exp_iQ(ϵ*pfield[ix,iy,iz,it])*gfield[μ][ix,iy,iz,it]
            end
        end
        end
        end
        end 
        return nothing
    end

    function updateP!(gfield::Gaugefield,pfield::Liefield,ϵ_hmc::Float64,fac::Float64)
        ϵ = ϵ_hmc*fac
        add_GaugeForce!(pfield,gfield,ϵ)
        return nothing
    end

    function add_GaugeForce!(pfield::Liefield,gfield::Gaugefield,ϵ::Float64)
        NX = gfield.NX
		NY = gfield.NY
		NZ = gfield.NZ
		NT = gfield.NT
        β = gfield.β

        @batch per=thread for it=1:NT
        for iz=1:NZ
        for iy=1:NY; iy_min = mod1(iy-1,NY); iy_plu = mod1(iy+1,NY); iz_min = mod1(iz-1,NZ); iz_plu = mod1(iz+1,NZ); it_min = mod1(it-1,NT); it_plu = mod1(it+1,NT);
        for ix=1:NX; ix_min = mod1(ix-1,NX); ix_plu = mod1(ix+1,NX); 
        staple= gfield[2][ix,iy    ,iz,    it]*gfield[1][ix,iy_plu,iz    ,it]*gfield[2][ix_plu,iy    ,iz    ,it    ]' +
                gfield[2][ix,iy_min,iz,    it]'*gfield[1][ix,iy_min,iz    ,it]*gfield[2][ix_plu,iy_min,iz    ,it    ] +
                gfield[3][ix,iy    ,iz    ,it]*gfield[1][ix,iy    ,iz_plu,it]*gfield[3][ix_plu,iy    ,iz    ,it    ]' +
                gfield[3][ix,iy    ,iz_min,it]'*gfield[1][ix,iy    ,iz_min,it]*gfield[3][ix_plu,iy    ,iz_min,it    ] +
                gfield[4][ix,iy    ,iz    ,it]*gfield[1][ix,iy    ,iz,it_plu]*gfield[4][ix_plu,iy    ,iz    ,it    ]' +
                gfield[4][ix,iy    ,iz,it_min]'*gfield[1][ix,iy    ,iz,it_min]*gfield[4][ix_plu,iy    ,iz    ,it_min]
        force = -β/12*im*( gfield[1][ix,iy,iz,it]*staple' - staple*gfield[1][ix,iy,iz,it] )
        pfield[ix,iy,iz,it] -= ϵ*force

        staple= gfield[1][ix,iy    ,iz,    it]*gfield[2][ix_plu,iy,iz    ,it    ]*gfield[1][ix,iy_plu,iz    ,it    ]' +
                gfield[1][ix_min,iy,iz,it    ]'*gfield[2][ix_min,iy,iz    ,it    ]*gfield[1][ix_min,iy_plu,iz,it    ] +
                gfield[3][ix,iy    ,iz    ,it]*gfield[2][ix,iy    ,iz_plu,it    ]*gfield[3][ix,iy_plu,iz    ,it    ]' +
                gfield[3][ix,iy    ,iz_min,it]'*gfield[2][ix,iy    ,iz_min,it    ]*gfield[3][ix,iy_plu,iz_min,it    ] +
                gfield[4][ix,iy    ,iz    ,it]*gfield[2][ix,iy    ,iz    ,it_plu]*gfield[4][ix,iy_plu,iz    ,it    ]' +
                gfield[4][ix,iy    ,iz,it_min]'*gfield[2][ix,iy    ,iz    ,it_min]*gfield[4][ix,iy_plu,iz    ,it_min]
        force = -β/12*im*( gfield[2][ix,iy,iz,it]*staple' - staple'*gfield[2][ix,iy,iz,it] )
        pfield[ix,iy,iz,it] -= ϵ*force

        staple= gfield[1][ix,iy    ,iz,    it]*gfield[3][ix_plu,iy,iz    ,it    ]*gfield[1][ix,iy    ,iz_plu,it    ]' +
                gfield[1][ix_min,iy,iz,    it]'*gfield[3][ix_min,iy,iz    ,it    ]*gfield[1][ix_min,iy,iz_plu,it    ] +
                gfield[2][ix,iy    ,iz    ,it]*gfield[3][ix,iy_plu,iz    ,it    ]*gfield[2][ix,iy    ,iz_plu,it    ]' +
                gfield[2][ix,iy_min,iz    ,it]'*gfield[3][ix,iy_min,iz    ,it    ]*gfield[2][ix,iy_min,iz_plu,it    ] +
                gfield[4][ix,iy    ,iz    ,it]*gfield[3][ix,iy    ,iz    ,it_plu]*gfield[4][ix,iy    ,iz_plu,it    ]' +
                gfield[4][ix,iy    ,iz,it_min]'*gfield[3][ix,iy    ,iz    ,it_min]*gfield[4][ix,iy    ,iz_plu,it_min]
        force = -β/12*im*( gfield[3][ix,iy,iz,it]*staple' - staple'*gfield[3][ix,iy,iz,it] )
        pfield[ix,iy,iz,it] -= ϵ*force

        staple= gfield[1][ix,iy    ,iz,    it]*gfield[4][ix_plu,iy,iz    ,it    ]*gfield[1][ix,iy    ,iz    ,it_plu]' +
                gfield[1][ix_min,iy,iz,    it]'*gfield[4][ix_min,iy,iz    ,it    ]*gfield[1][ix_min,iy_min,iz,it_plu] +
                gfield[2][ix,iy    ,iz    ,it]*gfield[4][ix,iy_plu,iz    ,it    ]*gfield[2][ix,iy    ,iz    ,it_plu]' +
                gfield[2][ix,iy_min,iz    ,it]'*gfield[4][ix,iy_min,iz    ,it    ]*gfield[2][ix,iy_min,iz    ,it_plu] +
                gfield[3][ix,iy    ,iz    ,it]*gfield[4][ix,iy    ,iz_plu,it    ]*gfield[3][ix,iy    ,iz    ,it_plu]' +
                gfield[3][ix,iy    ,iz_min,it]'*gfield[4][ix,iy    ,iz_min,it    ]*gfield[3][ix,iy    ,iz_min,it_plu]
        force = -β/12*im*( gfield[4][ix,iy,iz,it]*staple' - staple'*gfield[4][ix,iy,iz,it] )
        pfield[ix,iy,iz,it] -= ϵ*force
        end
        end
        end
        end
		return nothing
	end

end
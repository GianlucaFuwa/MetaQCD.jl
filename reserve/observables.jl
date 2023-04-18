module Observables 
    using LinearAlgebra
    using StaticArrays
	#using Base.Threads
	using Polyester
 
	import ..System_parameters: Params
	import ..Utils: exp_iQ
    import ..Gaugefields: Site_coords,unit_vector,Gaugefield,clover_sum
	import ..Stout_smearing: stout_smear!

	function polyakov_tr(g::Gaugefield,origin::Site_coords)
		ix = origin.x
		iy = origin.y
		iz = origin.z
		it = origin.t
		NT = g.NT
		poly = g[4][origin]
		for t=1:NT-1
			poly *= g[4][ix,iy,iz,mod1(it+t,NT)]
		end
		return tr(poly)
	end 

end

function top_charge_plaq1(g::Gaugefield)
    space = 8
    NX, NY, NZ, NT = size(g)
    Qplaq = zeros(Float64, nthreads()*space)

    @batch for it = 1:NT; it_plu = mod1(it+1, NT)
        for iz = 1:NZ; iz_plu = mod1(iz+1, NZ)
            for iy = 1:NY; iy_plu = mod1(iy+1, NY)
                for ix = 1:NX; ix_plu = mod1(ix+1, NX);
                    C12 = g[1][ix,iy,iz,it]  * g[2][ix_plu,iy,iz,it]  * g[1][ix,iy_plu,iz,it]' * g[2][ix,iy,iz,it]'
                    C12 = -im * (C12 - C12')

                    C13 = g[1][ix,iy,iz,it]  * g[3][ix_plu,iy,iz,it]  * g[1][ix,iy,iz_plu,it]' * g[3][ix,iy,iz,it]'
                    C13 = -im * (C13 - C13')

                    C23 = g[2][ix,iy,iz,it]  * g[3][ix,iy_plu,iz,it]  * g[2][ix,iy,iz_plu,it]' * g[3][ix,iy,iz,it]'
                    C23 = -im * (C23 - C23')

                    C14 = g[1][ix,iy,iz,it]  * g[4][ix_plu,iy,iz,it]  * g[1][ix,iy,iz,it_plu]' * g[4][ix,iy,iz,it]'
                    C14 = -im * (C14 - C14')

                    C24 = g[2][ix,iy,iz,it]  * g[4][ix,iy_plu,iz,it]  * g[2][ix,iy,iz,it_plu]' * g[4][ix,iy,iz,it]'
                    C24 = -im * (C24 - C24')

                    C34 = g[3][ix,iy,iz,it]  * g[4][ix,iy,iz_plu,it]  * g[3][ix,iy,iz,it_plu]' * g[4][ix,iy,iz,it]'
                    C34 = -im * (C34 - C34')
                    Qplaq[threadid()*space] += real(tr( C14*C23 + C13*C24 + C12*C34 ))
                end
            end
        end
    end
    return -1/8π^2 * sum(Qplaq)
end

function top_charge_clover1(g::Gaugefield)
    space = 8
    NX,NY,NZ,NT = size(g)
    Qclover = zeros(Float64,nthreads()*space)

    @batch for it=1:NT; it_min = mod1(it-1,NT); it_plu = mod1(it+1,NT)
        for iz=1:NZ; iz_min = mod1(iz-1,NZ); iz_plu = mod1(iz+1,NZ)
            for iy=1:NY; iy_min = mod1(iy-1,NY); iy_plu = mod1(iy+1,NY)
                for ix=1:NX; ix_min = mod1(ix-1,NX); ix_plu = mod1(ix+1,NX);
                    C12=g[1][ix    ,iy    ,iz,it]  * g[2][ix_plu,iy    ,iz,it]  * g[1][ix    ,iy_plu,iz,it]' * g[2][ix    ,iy    ,iz,it]' +
                        g[2][ix    ,iy    ,iz,it]  * g[1][ix_min,iy_plu,iz,it]' * g[2][ix_min,iy    ,iz,it]' * g[1][ix_min,iy    ,iz,it]  + 
                        g[1][ix_min,iy    ,iz,it]' * g[2][ix_min,iy_min,iz,it]' * g[1][ix_min,iy_min,iz,it]  * g[2][ix    ,iy_min,iz,it]  +
                        g[2][ix    ,iy_min,iz,it]' * g[1][ix    ,iy_min,iz,it]  * g[2][ix_plu,iy_min,iz,it]  * g[1][ix    ,iy    ,iz,it]'
                    C12 = -im * ( (C12 - C12') - 1/3*tr(C12 - C12')*I )

                    C13=g[1][ix    ,iy,iz    ,it]  * g[3][ix_plu,iy,iz    ,it]  * g[1][ix    ,iy,iz_plu,it]' * g[3][ix    ,iy,iz    ,it]' +
                        g[3][ix    ,iy,iz    ,it]  * g[1][ix_min,iy,iz_plu,it]' * g[3][ix_min,iy,iz    ,it]' * g[1][ix_min,iy,iz    ,it]  + 
                        g[1][ix_min,iy,iz    ,it]' * g[3][ix_min,iy,iz_min,it]' * g[1][ix_min,iy,iz_min,it]  * g[3][ix    ,iy,iz_min,it]  +
                        g[3][ix    ,iy,iz_min,it]' * g[1][ix    ,iy,iz_min,it]  * g[3][ix_plu,iy,iz_min,it]  * g[1][ix    ,iy,iz    ,it]'
                    C13 = -im * ( (C13 - C13') - 1/3*tr(C13 - C13')*I )   
                    
                    C23=g[2][ix,iy    ,iz    ,it]  * g[3][ix,iy_plu,iz    ,it]  * g[2][ix,iy    ,iz_plu,it]' * g[3][ix,iy    ,iz    ,it]' +
                        g[3][ix,iy    ,iz    ,it]  * g[2][ix,iy_min,iz_plu,it]' * g[3][ix,iy_min,iz    ,it]' * g[2][ix,iy_min,iz    ,it]  + 
                        g[2][ix,iy_min,iz    ,it]' * g[3][ix,iy_min,iz_min,it]' * g[2][ix,iy_min,iz_min,it]  * g[3][ix,iy    ,iz_min,it]  +
                        g[3][ix,iy    ,iz_min,it]' * g[2][ix,iy    ,iz_min,it]  * g[3][ix,iy_plu,iz_min,it]  * g[2][ix,iy    ,iz    ,it]'
                    C23 = -im * ( (C23 - C23') - 1/3*tr(C23 - C23')*I )

                    C14=g[1][ix    ,iy,iz,it    ]  * g[4][ix_plu,iy,iz,it    ]  * g[1][ix    ,iy,iz,it_plu]' * g[4][ix    ,iy,iz,it    ]' +
                        g[4][ix    ,iy,iz,it    ]  * g[1][ix_min,iy,iz,it_plu]' * g[4][ix_min,iy,iz,it    ]' * g[1][ix_min,iy,iz,it    ]  + 
                        g[1][ix_min,iy,iz,it    ]' * g[4][ix_min,iy,iz,it_min]' * g[1][ix_min,iy,iz,it_min]  * g[4][ix    ,iy,iz,it_min]  +
                        g[4][ix    ,iy,iz,it_min]' * g[1][ix    ,iy,iz,it_min]  * g[4][ix_plu,iy,iz,it_min]  * g[1][ix    ,iy,iz,it    ]'
                    C14 = -im * ( (C14 - C14') - 1/3*tr(C14 - C14')*I )

                    C24=g[2][ix,iy    ,iz,it    ]  * g[4][ix,iy_plu,iz,it    ]  * g[2][ix,iy    ,iz,it_plu]' * g[4][ix,iy    ,iz,it    ]' +
                        g[4][ix,iy    ,iz,it    ]  * g[2][ix,iy_min,iz,it_plu]' * g[4][ix,iy_min,iz,it    ]' * g[2][ix,iy_min,iz,it    ]  + 
                        g[2][ix,iy_min,iz,it    ]' * g[4][ix,iy_min,iz,it_min]' * g[2][ix,iy_min,iz,it_min]  * g[4][ix,iy    ,iz,it_min]  +
                        g[4][ix,iy    ,iz,it_min]' * g[2][ix,iy    ,iz,it_min]  * g[4][ix,iy_plu,iz,it_min]  * g[2][ix,iy    ,iz,it    ]'
                    C24 = -im * ( (C24 - C24') - 1/3*tr(C24 - C24')*I )

                    C34=g[3][ix,iy,iz    ,it    ]  * g[4][ix,iy,iz_plu,it    ]  * g[3][ix,iy,iz    ,it_plu]' * g[4][ix,iy,iz    ,it    ]' +
                        g[4][ix,iy,iz    ,it    ]  * g[3][ix,iy,iz_min,it_plu]' * g[4][ix,iy,iz_min,it    ]' * g[3][ix,iy,iz_min,it    ]  +
                        g[3][ix,iy,iz_min,it    ]' * g[4][ix,iy,iz_min,it_min]' * g[3][ix,iy,iz_min,it_min]  * g[4][ix,iy,iz    ,it_min]  +
                        g[4][ix,iy,iz    ,it_min]' * g[3][ix,iy,iz    ,it_min]  * g[4][ix,iy,iz_plu,it_min]  * g[3][ix,iy,iz    ,it    ]'
                    C34 = -im * ( (C34 - C34') - 1/3*tr(C34 - C34')*I )
                    Qclover[threadid()*space] += -1/256π^2 * real(tr( C23*C14 + C13*C24 + C12*C34 ))
                end
            end
        end
    end
    return sum(Qclover)
end
function apply_stout_smearing!(Uout::Gaugefield, U::Gaugefield, ρs::Vector{Float64})
    NX,NY,NZ,NT = size(U)
    ρ_stout = get_ρs(layer)
    staples = similar(U)
    staple_eachsite!(staples,U)
    
    @batch per=thread for it=1:NT
        for iz=1:NZ
            for iy=1:NY; iy_min = mod1(iy-1,NY); iy_plu = mod1(iy+1,NY);iz_min = mod1(iz-1,NZ); iz_plu = mod1(iz+1,NZ);it_min = mod1(it-1,NT); it_plu = mod1(it+1,NT)
                for ix=1:NX; ix_min = mod1(ix-1,NX); ix_plu = mod1(ix+1,NX);
                    tmp=U[2][ix,iy    ,iz,    it] * U[1][ix,iy_plu,iz    ,it] * U[2][ix_plu,iy    ,iz    ,it    ]' +
                        U[2][ix,iy_min,iz,    it]' * U[1][ix,iy_min,iz    ,it] * U[2][ix_plu,iy_min,iz    ,it    ] +
                        U[3][ix,iy    ,iz    ,it] * U[1][ix,iy    ,iz_plu,it] * U[3][ix_plu,iy    ,iz    ,it    ]' +
                        U[3][ix,iy    ,iz_min,it]' * U[1][ix,iy    ,iz_min,it] * U[3][ix_plu,iy    ,iz_min,it    ] +
                        U[4][ix,iy    ,iz    ,it] * U[1][ix,iy    ,iz,it_plu] * U[4][ix_plu,iy    ,iz    ,it    ]' +
                        U[4][ix,iy    ,iz,it_min]' * U[1][ix,iy    ,iz,it_min] * U[4][ix_plu,iy    ,iz    ,it_min]
                    Ω = tmp * ρs[1] * U[1][ix,iy,iz,it]'
                    Q = 0.5im*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
                    Uout[1][ix,iy,iz,it] = exp_iQ(Q) * U[1][ix,iy,iz,it]

                    tmp=U[1][ix,iy    ,iz,    it] * U[2][ix_plu,iy,iz    ,it    ] * U[1][ix,iy_plu,iz    ,it    ]' +
                        U[1][ix_min,iy,iz,it    ]' * U[2][ix_min,iy,iz    ,it    ] * U[1][ix_min,iy_plu,iz,it    ] +
                        U[3][ix,iy    ,iz    ,it] * U[2][ix,iy    ,iz_plu,it    ] * U[3][ix,iy_plu,iz    ,it    ]' +
                        U[3][ix,iy    ,iz_min,it]' * U[2][ix,iy    ,iz_min,it    ] * U[3][ix,iy_plu,iz_min,it    ] +
                        U[4][ix,iy    ,iz    ,it] * U[2][ix,iy    ,iz    ,it_plu] * U[4][ix,iy_plu,iz    ,it    ]' +
                        U[4][ix,iy    ,iz,it_min]' * U[2][ix,iy    ,iz    ,it_min] * U[4][ix,iy_plu,iz    ,it_min]
                    Ω = tmp * ρs[2] * U[2][ix,iy,iz,it]'
                    Q = 0.5im*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
                    Uout[2][ix,iy,iz,it] = exp_iQ(Q) * U[2][ix,iy,iz,it]

                    tmp=U[1][ix,iy    ,iz,    it] * U[3][ix_plu,iy,iz    ,it    ] * U[1][ix,iy    ,iz_plu,it    ]' +
                        U[1][ix_min,iy,iz,    it]' * U[3][ix_min,iy,iz    ,it    ] * U[1][ix_min,iy,iz_plu,it    ] +
                        U[2][ix,iy    ,iz    ,it] * U[3][ix,iy_plu,iz    ,it    ] * U[2][ix,iy    ,iz_plu,it    ]' +
                        U[2][ix,iy_min,iz    ,it]' * U[3][ix,iy_min,iz    ,it    ] * U[2][ix,iy_min,iz_plu,it    ] +
                        U[4][ix,iy    ,iz    ,it] * U[3][ix,iy    ,iz    ,it_plu] * U[4][ix,iy    ,iz_plu,it    ]' +
                        U[4][ix,iy    ,iz,it_min]' * U[3][ix,iy    ,iz    ,it_min] * U[4][ix,iy    ,iz_plu,it_min]
                    Ω = tmp * ρs[3] * U[3][ix,iy,iz,it]'
                    Q = 0.5im*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
                    Uout[3][ix,iy,iz,it] = exp_iQ(Q) * U[3][ix,iy,iz,it]

                    tmp=U[1][ix,iy    ,iz,    it] * U[4][ix_plu,iy,iz    ,it    ] * U[1][ix,iy    ,iz    ,it_plu]' +
                        U[1][ix_min,iy,iz,    it]' * U[4][ix_min,iy,iz    ,it    ] * U[1][ix_min,iy_min,iz,it_plu] +
                        U[2][ix,iy    ,iz    ,it] * U[4][ix,iy_plu,iz    ,it    ] * U[2][ix,iy    ,iz    ,it_plu]' +
                        U[2][ix,iy_min,iz    ,it]' * U[4][ix,iy_min,iz    ,it    ] * U[2][ix,iy_min,iz    ,it_plu] +
                        U[3][ix,iy    ,iz    ,it] * U[4][ix,iy    ,iz_plu,it    ] * U[3][ix,iy    ,iz    ,it_plu]' +
                        U[3][ix,iy    ,iz_min,it]' * U[4][ix,iy    ,iz_min,it    ] * U[3][ix,iy    ,iz_min,it_plu]
                    Ω = tmp * ρs[4] * U[4][ix,iy,iz,it]'
                    Q = 0.5im*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
                    Uout[4][ix,iy,iz,it] = exp_iQ(Q) * U[4][ix,iy,iz,it]
                end
            end
        end
    end
    
    return nothing
end

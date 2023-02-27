#
using Distributed
@everywhere using Base.Threads
@everywhere using StaticArrays
@everywhere using LinearAlgebra
@everywhere using Random
@everywhere using TimerOutputs
@everywhere using BenchmarkTools
@everywhere using Polyester
const to = TimerOutput()

U = rand(ComplexF64,3,3,4,16,16,16,16);

U2 = Array{Array{ComplexF64,6},1}(undef,0)
push!(U2,U[:,:,1,:,:,:,:])
push!(U2,U[:,:,2,:,:,:,:])
push!(U2,U[:,:,3,:,:,:,:])
push!(U2,U[:,:,4,:,:,:,:])

function vecofmat(U)
    U2 = Array{Array{SMatrix{3,3,ComplexF64,9},4},1}(undef,0)
    for μ=1:4
        Uμ = Array{SMatrix{3,3,ComplexF64,9},4}(undef,16,16,16,16)
        for t=1:4
            for z=1:4
                for y=1:4
                    for x=1:4
                        Uμ[x,y,z,t] = U[:,:,μ,x,y,z,t]
                    end
                end
            end
        end
    push!(U2,Uμ)
    end
    return U2
end
U3 = vecofmat(U)
#
@everywhere function darts_in_circle(N)
    n = zeros(nthreads()*8)
    @threads for i in 1:N
        if rand()^2+rand()^2<1
            n[threadid()*8]+=1
        end
    end
    return sum(n)
end

function pi_dist(N,loops)
    n = sum(pmap((x)->darts_in_circle(N),1:loops))
    return 4*n/(loops*N)
end

function stout_smear!(Usmr::Array{ComplexF64,7},ρ_stout::Float64)
    NX = size(Usmr,4)
    NY = size(Usmr,5)
    NZ = size(Usmr,6)
    NT = size(Usmr,7)
    U = deepcopy(Usmr)
    tmp = zeros(ComplexF64,3,3)

    @batch for it=1:NT
    for iz=1:NZ
    for iy=1:NY; iy_min = mod1(iy-1,NY); iy_plu = mod1(iy+1,NY);iz_min = mod1(iz-1,NZ); iz_plu = mod1(iz+1,NZ);it_min = mod1(it-1,NT); it_plu = mod1(it+1,NT)
    for ix=1:NX; ix_min = mod1(ix-1,NX); ix_plu = mod1(ix+1,NX);
        tmp=U[:,:,2,ix,iy    ,iz,    it]*U[:,:,1,ix,iy_plu,iz    ,it]*U[:,:,2,ix_plu,iy    ,iz    ,it    ] +
            U[:,:,2,ix,iy_min,iz,    it]*U[:,:,1,ix,iy_min,iz    ,it]*U[:,:,2,ix_plu,iy_min,iz    ,it    ] +
            U[:,:,3,ix,iy    ,iz    ,it]*U[:,:,1,ix,iy    ,iz_plu,it]*U[:,:,3,ix_plu,iy    ,iz    ,it    ] +
            U[:,:,3,ix,iy    ,iz_min,it]*U[:,:,1,ix,iy    ,iz_min,it]*U[:,:,3,ix_plu,iy    ,iz_min,it    ] +
            U[:,:,4,ix,iy    ,iz    ,it]*U[:,:,1,ix,iy    ,iz,it_plu]*U[:,:,4,ix_plu,iy    ,iz    ,it    ] +
            U[:,:,4,ix,iy    ,iz,it_min]*U[:,:,1,ix,iy    ,iz,it_min]*U[:,:,4,ix_plu,iy    ,iz    ,it_min]
        Ω = tmp*ρ_stout*U[:,:,1,ix,iy,iz,it]'
        Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
        Usmr[:,:,1,ix,iy,iz,it] = exp(SMatrix{3,3,ComplexF64,9}(Q))*Usmr[:,:,1,ix,iy,iz,it]

        tmp=U[:,:,1,ix,iy    ,iz,    it]*U[:,:,2,ix_plu,iy,iz    ,it    ]*U[:,:,1,ix,iy_plu,iz    ,it    ] +
            U[:,:,1,ix_min,iy,iz,it    ]*U[:,:,2,ix_min,iy,iz    ,it    ]*U[:,:,1,ix_min,iy_plu,iz,it    ] +
            U[:,:,3,ix,iy    ,iz    ,it]*U[:,:,2,ix,iy    ,iz_plu,it    ]*U[:,:,3,ix,iy_plu,iz    ,it    ] +
            U[:,:,3,ix,iy    ,iz_min,it]*U[:,:,2,ix,iy    ,iz_min,it    ]*U[:,:,3,ix,iy_plu,iz_min,it    ] +
            U[:,:,4,ix,iy    ,iz    ,it]*U[:,:,2,ix,iy    ,iz    ,it_plu]*U[:,:,4,ix,iy_plu,iz    ,it    ] +
            U[:,:,4,ix,iy    ,iz,it_min]*U[:,:,2,ix,iy    ,iz    ,it_min]*U[:,:,4,ix,iy_plu,iz    ,it_min]
        Ω = tmp*ρ_stout*U[:,:,2,ix,iy,iz,it]'
        Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
        Usmr[:,:,2,ix,iy,iz,it] = exp(SMatrix{3,3,ComplexF64,9}(Q))*Usmr[:,:,2,ix,iy,iz,it]

        tmp=U[:,:,1,ix,iy    ,iz,    it]*U[:,:,3,ix_plu,iy,iz    ,it    ]*U[:,:,1,ix,iy    ,iz_plu,it    ] +
            U[:,:,1,ix_min,iy,iz,    it]*U[:,:,3,ix_min,iy,iz    ,it    ]*U[:,:,1,ix_min,iy,iz_plu,it    ] +
            U[:,:,2,ix,iy    ,iz    ,it]*U[:,:,3,ix,iy_plu,iz    ,it    ]*U[:,:,2,ix,iy    ,iz_plu,it    ] +
            U[:,:,2,ix,iy_min,iz    ,it]*U[:,:,3,ix,iy_min,iz    ,it    ]*U[:,:,2,ix,iy_min,iz_plu,it    ] +
            U[:,:,4,ix,iy    ,iz    ,it]*U[:,:,3,ix,iy    ,iz    ,it_plu]*U[:,:,4,ix,iy    ,iz_plu,it    ] +
            U[:,:,4,ix,iy    ,iz,it_min]*U[:,:,3,ix,iy    ,iz    ,it_min]*U[:,:,4,ix,iy    ,iz_plu,it_min]
        Ω = tmp*ρ_stout*U[:,:,3,ix,iy,iz,it]'
        Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
        Usmr[:,:,3,ix,iy,iz,it] = exp(SMatrix{3,3,ComplexF64,9}(Q))*SMatrix{3,3,ComplexF64,9}(Usmr[:,:,3,ix,iy,iz,it])

        tmp=U[:,:,1,ix,iy    ,iz,    it]*U[:,:,4,ix_plu,iy,iz    ,it    ]*U[:,:,1,ix,iy    ,iz    ,it_plu] +
            U[:,:,1,ix_min,iy,iz,    it]*U[:,:,4,ix_min,iy,iz    ,it    ]*U[:,:,1,ix_min,iy_min,iz,it_plu] +
            U[:,:,2,ix,iy    ,iz    ,it]*U[:,:,4,ix,iy_plu,iz    ,it    ]*U[:,:,2,ix,iy    ,iz    ,it_plu] +
            U[:,:,2,ix,iy_min,iz    ,it]*U[:,:,4,ix,iy_min,iz    ,it    ]*U[:,:,2,ix,iy_min,iz    ,it_plu] +
            U[:,:,3,ix,iy    ,iz    ,it]*U[:,:,4,ix,iy    ,iz_plu,it    ]*U[:,:,3,ix,iy    ,iz    ,it_plu] +
            U[:,:,3,ix,iy    ,iz_min,it]*U[:,:,4,ix,iy    ,iz_min,it    ]*U[:,:,3,ix,iy    ,iz_min,it_plu]
        Ω = tmp*ρ_stout*U[:,:,4,ix,iy,iz,it]'
        Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
        Usmr[:,:,4,ix,iy,iz,it] = exp(SMatrix{3,3,ComplexF64,9}(Q))*Usmr[:,:,4,ix,iy,iz,it]
    end
    end
    end
    end
    return nothing
end

U_old = deepcopy(U3)

function stout_smear3!(Usmr::Array{Array{SMatrix{3,3,ComplexF64,9},4},1},U::Array{Array{SMatrix{3,3,ComplexF64,9},4},1},ρ_stout::Float64)
    NX = size(Usmr[1],1)
    NY = size(Usmr[1],2)
    NZ = size(Usmr[1],3)
    NT = size(Usmr[1],4)

    @batch per=thread for it=1:NT; it_min = mod1(it-1,NT); it_plu = mod1(it+1,NT)
    for iz=1:NZ; iz_min = mod1(iz-1,NZ); iz_plu = mod1(iz+1,NZ)
    for iy=1:NY; iy_min = mod1(iy-1,NY); iy_plu = mod1(iy+1,NY)
    for ix=1:NX; ix_min = mod1(ix-1,NX); ix_plu = mod1(ix+1,NX)
        tmp=U[2][ix,iy    ,iz,    it]*U[1][ix,iy_plu,iz    ,it]*U[2][ix_plu,iy    ,iz    ,it    ]' +
            U[2][ix,iy_min,iz,    it]'*U[1][ix,iy_min,iz    ,it]*U[2][ix_plu,iy_min,iz    ,it    ] +
            U[3][ix,iy    ,iz    ,it]*U[1][ix,iy    ,iz_plu,it]*U[3][ix_plu,iy    ,iz    ,it    ]' +
            U[3][ix,iy    ,iz_min,it]'*U[1][ix,iy    ,iz_min,it]*U[3][ix_plu,iy    ,iz_min,it    ] +
            U[4][ix,iy    ,iz    ,it]*U[1][ix,iy    ,iz,it_plu]*U[4][ix_plu,iy    ,iz    ,it    ]' +
            U[4][ix,iy    ,iz,it_min]'*U[1][ix,iy    ,iz,it_min]*U[4][ix_plu,iy    ,iz    ,it_min]
        Ω = tmp*ρ_stout*U[1][ix,iy,iz,it]'
        Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
        Usmr[1][ix,iy,iz,it] = exp(Q)*Usmr[1][ix,iy,iz,it]

        tmp=U[1][ix,iy    ,iz,    it]*U[2][ix_plu,iy,iz    ,it    ]*U[1][ix,iy_plu,iz    ,it    ]' +
            U[1][ix_min,iy,iz,it    ]'*U[2][ix_min,iy,iz    ,it    ]*U[1][ix_min,iy_plu,iz,it    ] +
            U[3][ix,iy    ,iz    ,it]*U[2][ix,iy    ,iz_plu,it    ]*U[3][ix,iy_plu,iz    ,it    ]' +
            U[3][ix,iy    ,iz_min,it]'*U[2][ix,iy    ,iz_min,it    ]*U[3][ix,iy_plu,iz_min,it    ] +
            U[4][ix,iy    ,iz    ,it]*U[2][ix,iy    ,iz    ,it_plu]*U[4][ix,iy_plu,iz    ,it    ]' +
            U[4][ix,iy    ,iz,it_min]'*U[2][ix,iy    ,iz    ,it_min]*U[4][ix,iy_plu,iz    ,it_min]
        Ω = tmp*ρ_stout*U[2][ix,iy,iz,it]'
        Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
        Usmr[2][ix,iy,iz,it] = exp(Q)*Usmr[2][ix,iy,iz,it]

        tmp=U[1][ix,iy    ,iz,    it]*U[3][ix_plu,iy,iz    ,it    ]*U[1][ix,iy    ,iz_plu,it    ]' +
            U[1][ix_min,iy,iz,    it]'*U[3][ix_min,iy,iz    ,it    ]*U[1][ix_min,iy,iz_plu,it    ] +
            U[2][ix,iy    ,iz    ,it]*U[3][ix,iy_plu,iz    ,it    ]*U[2][ix,iy    ,iz_plu,it    ]' +
            U[2][ix,iy_min,iz    ,it]'*U[3][ix,iy_min,iz    ,it    ]*U[2][ix,iy_min,iz_plu,it    ] +
            U[4][ix,iy    ,iz    ,it]*U[3][ix,iy    ,iz    ,it_plu]*U[4][ix,iy    ,iz_plu,it    ]' +
            U[4][ix,iy    ,iz,it_min]'*U[3][ix,iy    ,iz    ,it_min]*U[4][ix,iy    ,iz_plu,it_min]
        Ω = tmp*ρ_stout*U[3][ix,iy,iz,it]'
        Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
        Usmr[3][ix,iy,iz,it] = exp(Q)*Usmr[3][ix,iy,iz,it]

        tmp=U[1][ix,iy    ,iz,    it]*U[4][ix_plu,iy,iz    ,it    ]*U[1][ix,iy    ,iz    ,it_plu]' +
            U[1][ix_min,iy,iz,    it]'*U[4][ix_min,iy,iz    ,it    ]*U[1][ix_min,iy_min,iz,it_plu] +
            U[2][ix,iy    ,iz    ,it]*U[4][ix,iy_plu,iz    ,it    ]*U[2][ix,iy    ,iz    ,it_plu]' +
            U[2][ix,iy_min,iz    ,it]'*U[4][ix,iy_min,iz    ,it    ]*U[2][ix,iy_min,iz    ,it_plu] +
            U[3][ix,iy    ,iz    ,it]*U[4][ix,iy    ,iz_plu,it    ]*U[3][ix,iy    ,iz    ,it_plu]' +
            U[3][ix,iy    ,iz_min,it]'*U[4][ix,iy    ,iz_min,it    ]*U[3][ix,iy    ,iz_min,it_plu]
        Ω = tmp*ρ_stout*U[4][ix,iy,iz,it]'
        Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
        Usmr[4][ix,iy,iz,it] = exp(Q)*Usmr[4][ix,iy,iz,it]
    end
    end
    end
    end
    return nothing
end

function stout_smear2!(Usmr::Array{Array{SMatrix{3,3,ComplexF64,9},4},1},U::Array{Array{SMatrix{3,3,ComplexF64,9},4},1},ρ_stout::Float64)
    NX = size(Usmr[1],1)
    NY = size(Usmr[1],2)
    NZ = size(Usmr[1],3)
    NT = size(Usmr[1],4)

    @threads for it=1:NT; it_min = mod1(it-1,NT); it_plu = mod1(it+1,NT)
    for iz=1:NZ; iz_min = mod1(iz-1,NZ); iz_plu = mod1(iz+1,NZ)
    for iy=1:NY; iy_min = mod1(iy-1,NY); iy_plu = mod1(iy+1,NY)
    for ix=1:NX; ix_min = mod1(ix-1,NX); ix_plu = mod1(ix+1,NX)
        tmp=U[2][ix,iy    ,iz,    it]*U[1][ix,iy_plu,iz    ,it]*U[2][ix_plu,iy    ,iz    ,it    ]' +
            U[2][ix,iy_min,iz,    it]'*U[1][ix,iy_min,iz    ,it]*U[2][ix_plu,iy_min,iz    ,it    ] +
            U[3][ix,iy    ,iz    ,it]*U[1][ix,iy    ,iz_plu,it]*U[3][ix_plu,iy    ,iz    ,it    ]' +
            U[3][ix,iy    ,iz_min,it]'*U[1][ix,iy    ,iz_min,it]*U[3][ix_plu,iy    ,iz_min,it    ] +
            U[4][ix,iy    ,iz    ,it]*U[1][ix,iy    ,iz,it_plu]*U[4][ix_plu,iy    ,iz    ,it    ]' +
            U[4][ix,iy    ,iz,it_min]'*U[1][ix,iy    ,iz,it_min]*U[4][ix_plu,iy    ,iz    ,it_min]
        Ω = tmp*ρ_stout*U[1][ix,iy,iz,it]'
        Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
        Usmr[1][ix,iy,iz,it] = exp(Q)*Usmr[1][ix,iy,iz,it]

        tmp=U[1][ix,iy    ,iz,    it]*U[2][ix_plu,iy,iz    ,it    ]*U[1][ix,iy_plu,iz    ,it    ]' +
            U[1][ix_min,iy,iz,it    ]'*U[2][ix_min,iy,iz    ,it    ]*U[1][ix_min,iy_plu,iz,it    ] +
            U[3][ix,iy    ,iz    ,it]*U[2][ix,iy    ,iz_plu,it    ]*U[3][ix,iy_plu,iz    ,it    ]' +
            U[3][ix,iy    ,iz_min,it]'*U[2][ix,iy    ,iz_min,it    ]*U[3][ix,iy_plu,iz_min,it    ] +
            U[4][ix,iy    ,iz    ,it]*U[2][ix,iy    ,iz    ,it_plu]*U[4][ix,iy_plu,iz    ,it    ]' +
            U[4][ix,iy    ,iz,it_min]'*U[2][ix,iy    ,iz    ,it_min]*U[4][ix,iy_plu,iz    ,it_min]
        Ω = tmp*ρ_stout*U[2][ix,iy,iz,it]'
        Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
        Usmr[2][ix,iy,iz,it] = exp(Q)*Usmr[2][ix,iy,iz,it]

        tmp=U[1][ix,iy    ,iz,    it]*U[3][ix_plu,iy,iz    ,it    ]*U[1][ix,iy    ,iz_plu,it    ]' +
            U[1][ix_min,iy,iz,    it]'*U[3][ix_min,iy,iz    ,it    ]*U[1][ix_min,iy,iz_plu,it    ] +
            U[2][ix,iy    ,iz    ,it]*U[3][ix,iy_plu,iz    ,it    ]*U[2][ix,iy    ,iz_plu,it    ]' +
            U[2][ix,iy_min,iz    ,it]'*U[3][ix,iy_min,iz    ,it    ]*U[2][ix,iy_min,iz_plu,it    ] +
            U[4][ix,iy    ,iz    ,it]*U[3][ix,iy    ,iz    ,it_plu]*U[4][ix,iy    ,iz_plu,it    ]' +
            U[4][ix,iy    ,iz,it_min]'*U[3][ix,iy    ,iz    ,it_min]*U[4][ix,iy    ,iz_plu,it_min]
        Ω = tmp*ρ_stout*U[3][ix,iy,iz,it]'
        Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
        Usmr[3][ix,iy,iz,it] = exp(Q)*Usmr[3][ix,iy,iz,it]

        tmp=U[1][ix,iy    ,iz,    it]*U[4][ix_plu,iy,iz    ,it    ]*U[1][ix,iy    ,iz    ,it_plu]' +
            U[1][ix_min,iy,iz,    it]'*U[4][ix_min,iy,iz    ,it    ]*U[1][ix_min,iy_min,iz,it_plu] +
            U[2][ix,iy    ,iz    ,it]*U[4][ix,iy_plu,iz    ,it    ]*U[2][ix,iy    ,iz    ,it_plu]' +
            U[2][ix,iy_min,iz    ,it]'*U[4][ix,iy_min,iz    ,it    ]*U[2][ix,iy_min,iz    ,it_plu] +
            U[3][ix,iy    ,iz    ,it]*U[4][ix,iy    ,iz_plu,it    ]*U[3][ix,iy    ,iz    ,it_plu]' +
            U[3][ix,iy    ,iz_min,it]'*U[4][ix,iy    ,iz_min,it    ]*U[3][ix,iy    ,iz_min,it_plu]
        Ω = tmp*ρ_stout*U[4][ix,iy,iz,it]'
        Q = 1/2*(Ω'-Ω) - im/6*tr(Ω'-Ω)*I
        Usmr[4][ix,iy,iz,it] = exp(Q)*Usmr[4][ix,iy,iz,it]
    end
    end
    end
    end
    return nothing
end

function ncminrealtrace(U::SMatrix{3,3,ComplexF64,9})
    ncminrealtrace = Float64(0.0)
    ncminrealtrace = 3.0-real(tr(U))
    return ncminrealtrace
end

function calc_Sgwils4(U::Array{Array{SMatrix{3,3,ComplexF64,9},4},1})
    space = 8
    Sg = zeros(Float64,nthreads()*space)
    NX = size(U[1],1)
    NY = size(U[1],2)
    NZ = size(U[1],3)
    NT = size(U[1],4)
    @batch per=thread for it=1:NT
    for iz=1:NZ
    for iy=1:NY; iy_plu = mod1(iy+1,NY); iz_plu = mod1(iz+1,NZ); it_plu = mod1(it+1,NT);
    for ix=1:NX; ix_plu = mod1(ix+1,NX);
        Sg[threadid()*space] += ncminrealtrace(U[1][ix,iy,iz,it]*U[2][ix_plu,iy,iz,it]*U[1][ix,iy_plu,iz,it]'*U[2][ix,iy,iz,it]') +
            ncminrealtrace(U[2][ix,iy,iz,it]*U[3][ix,iy_plu,iz,it]*U[2][ix,iy,iz_plu,it]'*U[3][ix,iy,iz,it]') +
            ncminrealtrace(U[3][ix,iy,iz,it]*U[1][ix,iy,iz_plu,it]*U[3][ix_plu,iy,iz,it]'*U[1][ix,iy,iz,it]') +
            ncminrealtrace(U[1][ix,iy,iz,it]*U[4][ix_plu,iy,iz,it]*U[1][ix,iy,iz,it_plu]'*U[4][ix,iy,iz,it]') +
            ncminrealtrace(U[2][ix,iy,iz,it]*U[4][ix,iy_plu,iz,it]*U[2][ix,iy,iz,it_plu]'*U[4][ix,iy,iz,it]') +
            ncminrealtrace(U[3][ix,iy,iz,it]*U[4][ix,iy,iz_plu,it]*U[3][ix,iy,iz,it_plu]'*U[4][ix,iy,iz,it]')
    end
    end
    end
    end
    return sum(Sg)/3/(NX*NY*NZ*NT)
end

function calc_Sgwils(U::Array{ComplexF64,7})
    Sg = 0.0
    NX = size(U,4)
    NY = size(U,5)
    NZ = size(U,6)
    NT = size(U,7)
    for it=1:NT
    for iz=1:NZ
    for iy=1:NY; iy_plu = mod1(iy+1,NY); iz_plu = mod1(iz+1,NZ); it_plu = mod1(it+1,NT);
    for ix=1:NX; ix_plu = mod1(ix+1,NX);
        Sg += 3 - real(tr(U[:,:,1,ix,iy,iz,it]*U[:,:,2,ix_plu,iy,iz,it]*U[:,:,1,ix,iy_plu,iz,it]'*U[:,:,2,ix,iy,iz,it]')) +
              3 - real(tr(U[:,:,2,ix,iy,iz,it]*U[:,:,3,ix,iy_plu,iz,it]*U[:,:,2,ix,iy,iz_plu,it]'*U[:,:,3,ix,iy,iz,it]')) +
              3 - real(tr(U[:,:,3,ix,iy,iz,it]*U[:,:,1,ix,iy,iz_plu,it]*U[:,:,3,ix_plu,iy,iz,it]'*U[:,:,1,ix,iy,iz,it]')) +
              3 - real(tr(U[:,:,1,ix,iy,iz,it]*U[:,:,4,ix_plu,iy,iz,it]*U[:,:,1,ix,iy,iz,it_plu]'*U[:,:,4,ix,iy,iz,it]')) +
              3 - real(tr(U[:,:,2,ix,iy,iz,it]*U[:,:,4,ix,iy_plu,iz,it]*U[:,:,2,ix,iy,iz,it_plu]'*U[:,:,4,ix,iy,iz,it]')) +
              3 - real(tr(U[:,:,3,ix,iy,iz,it]*U[:,:,4,ix,iy,iz_plu,it]*U[:,:,3,ix,iy,iz,it_plu]'*U[:,:,4,ix,iy,iz,it]'))
    end
    end
    end
    end
    return Sg/3/(NX*NY*NZ*NT)
end

function calc_Sgwils2(U::Array{Array{ComplexF64,6},1})
    Sg = 0.0
    NX = size(U[1],3)
    NY = size(U[1],4)
    NZ = size(U[1],5)
    NT = size(U[1],6)
    for it=1:NT
    for iz=1:NZ
    for iy=1:NY; iy_plu = mod1(iy+1,NY); iz_plu = mod1(iz+1,NZ); it_plu = mod1(it+1,NT);
    for ix=1:NX; ix_plu = mod1(ix+1,NX);
        Sg += 3 - real(tr(U[1][:,:,ix,iy,iz,it]*U[2][:,:,ix_plu,iy,iz,it]*U[1][:,:,ix,iy_plu,iz,it]'*U[2][:,:,ix,iy,iz,it]')) +
              3 - real(tr(U[2][:,:,ix,iy,iz,it]*U[3][:,:,ix,iy_plu,iz,it]*U[2][:,:,ix,iy,iz_plu,it]'*U[3][:,:,ix,iy,iz,it]')) +
              3 - real(tr(U[3][:,:,ix,iy,iz,it]*U[1][:,:,ix,iy,iz_plu,it]*U[3][:,:,ix_plu,iy,iz,it]'*U[1][:,:,ix,iy,iz,it]')) +
              3 - real(tr(U[1][:,:,ix,iy,iz,it]*U[4][:,:,ix_plu,iy,iz,it]*U[1][:,:,ix,iy,iz,it_plu]'*U[4][:,:,ix,iy,iz,it]')) +
              3 - real(tr(U[2][:,:,ix,iy,iz,it]*U[4][:,:,ix,iy_plu,iz,it]*U[2][:,:,ix,iy,iz,it_plu]'*U[4][:,:,ix,iy,iz,it]')) +
              3 - real(tr(U[3][:,:,ix,iy,iz,it]*U[4][:,:,ix,iy,iz_plu,it]*U[3][:,:,ix,iy,iz,it_plu]'*U[4][:,:,ix,iy,iz,it]'))
    end
    end
    end
    end
    return Sg/3/(NX*NY*NZ*NT)
end

function calc_Sgwils3(U::Array{Array{SMatrix{3,3,ComplexF64,9},4},1})
    space = 8
    Sg = zeros(Float64,nthreads()*space)
    NX = size(U[1],1)
    NY = size(U[1],2)
    NZ = size(U[1],3)
    NT = size(U[1],4)
    NV = NX*NY*NZ*NT
    @threads for it=1:NT
    for iz=1:NZ
    for iy=1:NY; iy_plu = mod1(iy+1,NY); iz_plu = mod1(iz+1,NZ); it_plu = mod1(it+1,NT)
    for ix=1:NX; ix_plu = mod1(ix+1,NX)
        Sg[threadid()*space] += 3.0 - real(tr(U[1][ix,iy,iz,it]*U[2][ix_plu,iy,iz,it]*U[1][ix,iy_plu,iz,it]'*U[2][ix,iy,iz,it]')) +
              3.0 - real(tr(U[2][ix,iy,iz,it]*U[3][ix,iy_plu,iz,it]*U[2][ix,iy,iz_plu,it]'*U[3][ix,iy,iz,it]')) +
              3.0 - real(tr(U[3][ix,iy,iz,it]*U[1][ix,iy,iz_plu,it]*U[3][ix_plu,iy,iz,it]'*U[1][ix,iy,iz,it]')) +
              3.0 - real(tr(U[1][ix,iy,iz,it]*U[4][ix_plu,iy,iz,it]*U[1][ix,iy,iz,it_plu]'*U[4][ix,iy,iz,it]')) +
              3.0 - real(tr(U[2][ix,iy,iz,it]*U[4][ix,iy_plu,iz,it]*U[2][ix,iy,iz,it_plu]'*U[4][ix,iy,iz,it]')) +
              3.0 - real(tr(U[3][ix,iy,iz,it]*U[4][ix,iy,iz_plu,it]*U[3][ix,iy,iz,it_plu]'*U[4][ix,iy,iz,it]'))
    end
    end
    end
    end
    return sum(Sg)/3/NV
end

function test_inner(U,rng)
    NX = size(U[1],1)
    NY = size(U[1],2)
    NZ = size(U[1],3)
    NT = size(U[1],4)
    @threads for it=1:NT
    for iz=1:NZ
    for iy=1:NY
    for ix=1:NX
    for μ=1:4
        col1 = rand(rng,3).-0.5 + im*(rand(rng,3).-0.5)
        col1 /= norm(col1)
        col2 = rand(rng,3).-0.5 + im*(rand(rng,3).-0.5)
        col2 -= (col1'*col2)*col1
        col2 /= norm(col2)
        col3 = cross(conj(col1),conj(col2))
        U[μ][ix,iy,iz,it] = [col1 col2 col3]
    end
    end
    end
    end
    end
    return nothing
end

function test_outer(U,rng)
    NX = size(U[1],1)
    NY = size(U[1],2)
    NZ = size(U[1],3)
    NT = size(U[1],4)
    @threads for μ=1:4
    for it=1:NT
    for iz=1:NZ
    for iy=1:NY
    for ix=1:NX
        col1 = rand(rng,3).-0.5 + im*(rand(rng,3).-0.5)
        col1 /= norm(col1)
        col2 = rand(rng,3).-0.5 + im*(rand(rng,3).-0.5)
        col2 -= (col1'*col2)*col1
        col2 /= norm(col2)
        col3 = cross(conj(col1),conj(col2))
        U[μ][ix,iy,iz,it] = [col1 col2 col3]
    end
    end
    end
    end
    end
    return nothing
end
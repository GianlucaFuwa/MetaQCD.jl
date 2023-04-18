using Base.Threads
using StaticArrays
using LinearAlgebra
using Random
using TimerOutputs
using BenchmarkTools
using Polyester
const to = TimerOutput()
N=16
U = rand(ComplexF64,3,3,4,N,N,N,N);

U2 = Array{Array{ComplexF64,6},1}(undef,0)
push!(U2,U[:,:,1,:,:,:,:])
push!(U2,U[:,:,2,:,:,:,:])
push!(U2,U[:,:,3,:,:,:,:])
push!(U2,U[:,:,4,:,:,:,:])

function vecofmat(U)
    U2 = Array{Array{SMatrix{3,3,ComplexF64,9},4},1}(undef,0)
    for μ=1:4
        Uμ = Array{SMatrix{3,3,ComplexF64,9},4}(undef,N,N,N,N)
        for t=1:N
            for z=1:N
                for y=1:N
                    for x=1:N
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
function darts_in_circle(N)
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

struct Site_coords{T}
    x::T
    y::T
    z::T
    t::T
end 

function Base.:+(s::Site_coords,t::NTuple{4})
    x,y,z,t = (s.x,s.y,s.z,s.t) .+ t
    return x,y,z,t
end

function getcoords(s::Site_coords)
    return (s.x,s.y,s.z,s.t)
end

@inline function Base.getindex(U::Array{SMatrix{3,3,ComplexF64,9},4},s::Site_coords)
    x,y,z,t = getcoords(s)
    return U[x,y,z,t]
end

function unit_vector(μ::Int,steps::Int=1)
    return (steps*(μ==1),steps*(μ==2),steps*(μ==3),steps*(μ==4))
end
#=
function move!(s::Site_coords,stepsx,stepsy,stepsz,stepst)
    s.x += stepsx
    s.y += stepsy
    s.z += stepsz
    s.t += stepst
    return nothing
end

function move!(s::Site_coords,μ::Int64,steps::Int64,lim::Int64)
    if μ == 1
        s.x = mod1(s.x+steps,lim)
    elseif μ == 2
        s.y = mod1(s.y+steps,lim)
    elseif μ == 3
        s.z = mod1(s.z+steps,lim)
    elseif μ == 4
        s.t = mod1(s.t+steps,lim)
    end
    return nothing
end
=#
function move(s::Site_coords,μ::Int64,steps::Int64,lim::Int64)
    x,y,z,t = getcoords(s)
    if μ == 1
        x = mod1(x+steps,lim)
    elseif μ == 2
        y = mod1(y+steps,lim)
    elseif μ == 3
        z = mod1(z+steps,lim)
    elseif μ == 4
        t = mod1(t+steps,lim)
    end
    return Site_coords(x,y,z,t)
end

function calc_plaq(g::Array{Array{SMatrix{3,3,ComplexF64,9},4},1},μ::Int64,v::Int64,origin::Site_coords)
    NX = size(U[1],1)
    NY = size(U[1],2)
    NZ = size(U[1],3)
    NT = size(U[1],4)
    plaq = g[μ][origin] *g[v][move(origin,unit_vector(μ),(NX,NY,NZ,NT))] *
           g[μ][move(origin,unit_vector(v),(NX,NY,NZ,NT))]'*g[v][origin]'
    return plaq
end

function plaquette(g,μ::Int64,ν::Int64,ix,iy,iz,it) 
    NX,NY,NZ,NT = size(g[1])
    if μ == 4 && ν == 1
        it_plu = mod1(it+1,NT)
        ix_plu = mod1(ix+1,NX)
        plaq = g[4][ix,iy,iz,it]  * g[1][ix,iy,iz,it_plu]  * g[4][ix_plu,iy,iz,it]' * g[1][ix,iy,iz,it]'
    elseif μ == 4 && ν == 2
        it_plu = mod1(it+1,NT); iy_plu = mod1(iy+1,NY)
        plaq = g[4][ix,iy,iz,it]  * g[2][ix,iy,iz,it_plu]  * g[4][ix,iy_plu,iz,it]' * g[2][ix,iy,iz,it]'
    elseif μ == 4 && ν == 3
        it_plu = mod1(it+1,NT); iz_plu = mod1(iz+1,NZ)
        plaq = g[4][ix,iy,iz,it]  * g[3][ix,iy,iz,it_plu]  * g[4][ix,iy,iz_plu,it]' * g[3][ix,iy,iz,it]'
    elseif μ == 1 && ν == 2
        ix_plu = mod1(ix+1,NX); iy_plu = mod1(iy+1,NY)
        plaq = g[1][ix,iy,iz,it]  * g[2][ix_plu,iy,iz,it]  * g[1][ix,iy_plu,iz,it]' * g[2][ix,iy,iz,it]'
    elseif μ == 1 && ν == 3
        ix_plu = mod1(ix+1,NX); iz_plu = mod1(iz+1,NZ)
        plaq = g[1][ix,iy,iz,it]  * g[3][ix_plu,iy,iz,it]  * g[1][ix,iy,iz_plu,it]' * g[3][ix,iy,iz,it]'
    elseif μ == 2 && ν == 3  
        iy_plu = mod1(iy+1,NY); iz_plu = mod1(iz+1,NZ)
        plaq = g[2][ix,iy,iz,it]  * g[3][ix,iy_plu,iz,it]  * g[2][ix,iy,iz_plu,it]' * g[3][ix,iy,iz,it]'
    end
    return plaq
end

function plaquetteSC(g,μ::Int64,ν::Int64,s::Site_coords) 
    Nμ = size(g[1],μ)
    Nν = size(g[2],ν)
    plaq = g[μ][s] * g[ν][move(s,μ,1,Nμ)] * g[μ][move(s,ν,1,Nν)]' * g[ν][s]'
    return plaq
end

function plaquetteSC2(g,μ::Int64,ν::Int64,ix,iy,iz,it) 
    Nμ = size(g[1],μ)
    Nν = size(g[2],ν)
    c = Site_coords(ix,iy,iz,it)
    c1 = move(c,μ,1,Nμ)
    c2 = move(c,ν,1,Nν)
    plaq = g[μ][c] * g[ν][c1] * g[μ][c2]' * g[ν][c]'
    return plaq
end


function plaquette_trsum(g)
    space = 8
    plaq = zeros(ComplexF64,nthreads()*space)
    NX = size(g[1],1)
    NY = size(g[1],2)
    NZ = size(g[1],3)
    NT = size(g[1],4)
    @batch for it=1:NT
    for iz=1:NZ
    for iy=1:NY; iy_plu = mod1(iy+1,NY); iz_plu = mod1(iz+1,NZ); it_plu = mod1(it+1,NT);
    for ix=1:NX; ix_plu = mod1(ix+1,NX);
        plaq[threadid()*space] += tr(g[1][ix,iy,iz,it]*g[2][ix_plu,iy,iz,it]*g[1][ix,iy_plu,iz,it]'*g[2][ix,iy,iz,it]') +
              tr(g[2][ix,iy,iz,it]*g[3][ix,iy_plu,iz,it]*g[2][ix,iy,iz_plu,it]'*g[3][ix,iy,iz,it]') +
              tr(g[1][ix,iy,iz,it]*g[3][ix_plu,iy,iz,it]*g[1][ix,iy,iz_plu,it]'*g[3][ix,iy,iz,it]') +
              tr(g[1][ix,iy,iz,it]*g[4][ix_plu,iy,iz,it]*g[1][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]') +
              tr(g[2][ix,iy,iz,it]*g[4][ix,iy_plu,iz,it]*g[2][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]') +
              tr(g[3][ix,iy,iz,it]*g[4][ix,iy,iz_plu,it]*g[3][ix,iy,iz,it_plu]'*g[4][ix,iy,iz,it]')
    end
    end
    end
    end
    return sum(plaq)
end

function plaquette_trsumSC(g)
    space = 8
    plaq = zeros(ComplexF64,nthreads()*space)
    NX = size(g[1],1)
    NY = size(g[1],2)
    NZ = size(g[1],3)
    NT = size(g[1],4)
    @batch for it=1:NT
    for iz=1:NZ
    for iy=1:NY
    for ix=1:NX
        plaq[threadid()*space] += tr(plaquetteSC2(g,1,2,ix,iy,iz,it)) +
              tr(plaquetteSC2(g,1,3,ix,iy,iz,it)) +
              tr(plaquetteSC2(g,1,4,ix,iy,iz,it)) +
              tr(plaquetteSC2(g,2,3,ix,iy,iz,it)) +
              tr(plaquetteSC2(g,2,4,ix,iy,iz,it)) +
              tr(plaquetteSC2(g,3,4,ix,iy,iz,it))
    end
    end
    end
    end
    return sum(plaq)
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
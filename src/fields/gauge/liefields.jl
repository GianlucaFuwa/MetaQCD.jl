struct Liefield
    P::Vector{Array{SMatrix{3,3,ComplexF64,9},4}}
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NV::Int64
    NC::Int64	

    function Liefield(NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        P = Vector{Array{SMatrix{3,3,ComplexF64,9},4}}(undef, 0)

        for μ = 1:4
            Pμ = Array{SMatrix{3,3,ComplexF64,9},4}(undef, NX, NY, NZ, NT)
            fill!(Pμ, SMatrix{3,3,ComplexF64,9}(zeros(3, 3)))
            push!(P, Pμ)
        end

        return new(P, NX, NY, NZ, NT, NV, 3)
    end

    function Liefield(U::Gaugefield)
        return Liefield(U.NX, U.NY, U.NZ, U.NT)
    end
end

function Base.setindex!(p::Liefield, v, μ) 
    p.P[μ] = v
    return nothing
end

function Base.getindex(p::Liefield, μ)
    return p.P[μ]
end

function Base.size(p::Liefield)
    return p.NX, p.NY, p.NZ, p.NT
end

function clear!(p::Liefield)
    NX, NY, NZ, NT = size(p)

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        @inbounds p[μ][ix,iy,iz,it] = zeros(3, 3)
                    end
                end
            end
        end
    end

    return nothing
end

function gaussian_momenta!(p::Liefield, rng = Xoshiro())
    NX, NY, NZ, NT = size(p)
    sq3 = sqrt(3)

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        h = SVector{8,Float64}(randn(rng, 8))
                        p[μ][ix,iy,iz,it] = 0.5im * [
                            h[3]+h[8]/sq3  h[1]-im*h[2]   h[4]-im*h[5]
                            h[1]+im*h[2]  -h[3]+h[8]/sq3  h[6]-im*h[7]
                            h[4]+im*h[5]   h[6]+im*h[7]  -2*h[8]/sq3 
                        ]
                    end
                end
            end
        end
    end

    return nothing
end

function calc_kinetic_energy(p::Liefield)
    NX, NY, NZ, NT = size(p)
    space = 8
    #H_kin = zeros(nthreads() * space)
    H_kin = 0.0

    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for μ = 1:4
                        #H_kin[threadid() * space] += 
                        H_kin += 
                            real(tr(p[μ][ix,iy,iz,it], p[μ][ix,iy,iz,it]))
                    end
                end
            end
        end
    end 
    
    return sum(H_kin)
end
module Utils
    using LinearAlgebra
    using Random
    using StaticArrays

    function move(current_site::Site_coord,dir::Int;steps::Int=1)
        

    eye3 = SMatrix{3,3,ComplexF64}([
        1 0 0
        0 1 0
        0 0 1
    ])
    # Exponential function of iQ ∈ 𝔰𝔲(3), i.e. Projection of Q onto SU(3)
    # From Morningstar & Peardon (2008) arXiv:hep-lat/0311018v1
    function exp_iQ(Q::Union{SMatrix{3,3,ComplexF64,9},MMatrix{3,3,ComplexF64,9},Matrix{ComplexF64}})
        if norm(Q)>1e-10
            u,w = set_uw(Q)
            f0,f1,f2 = set_fj(u,w)
            return f0*I + f1*Q + f2*Q^2
        else
            return eye3
        end
    end

    function set_uw(Q::Union{SMatrix{3,3,ComplexF64,9},MMatrix{3,3,ComplexF64,9},Matrix{ComplexF64}})
        c0 = 1/3*tr(Q^3)
        c1 = 1/2*tr(Q^2)
        c13r = sqrt(c1/3)
        c0max = 2*c13r^3
        Θ = acos(c0/c0max)

        u = c13r*cos(Θ/3)
        w = sqrt(c1)*sin(Θ/3)
        return u,w
    end

    function set_fj(u,w)
        if w == 0
            ξ0 = 1
        else
            ξ0 = sin(w)/w
        end
        e2iu = exp(2im*u)
        emiu = exp(-im*u)
        cosw = cos(w)

        h0 = (u^2-w^2)*e2iu + emiu*(8*u^2*cosw+2im*u*(3u^2+w^2)ξ0)
        h1 = 2*u*e2iu - emiu*(2*u*cosw-im*(3u^2-w^2)*ξ0)
        h2 = e2iu - emiu*(cosw+3im*u*ξ0)
        
        fden = 1/(9u^2-w^2)
        f0 = h0*fden
        f1 = h1*fden
        f2 = h2*fden
        return f0,f1,f2
    end

    sigma = []
    s = SMatrix{2,2,ComplexF64}([
        0 1
        1 0
    ])
    push!(sigma,s)
    s = SMatrix{2,2,ComplexF64}([
        0 -im
        im 0
    ])
    push!(sigma,s)
    s = SMatrix{2,2,ComplexF64}([
        1 0
        0 -1
    ])
    push!(sigma,s)

    # Generator of local Update Proposal Matrices X
    # From Gattringer C. & Lang C.B. (Springer, Berlin Heidelberg 2010)
    function gen_proposal(rng::Xoshiro,ϵ::Float64)
        r = rand(rng,4).-0.5
        t = rand(rng,4).-0.5
        s = rand(rng,4).-0.5
        R2 = sign(r[1])*sqrt(1-ϵ^2)*I+im*(ϵ*r[2:end]'/norm(r[2:end])*sigma)
        S2 = sign(s[1])*sqrt(1-ϵ^2)*I+im*(ϵ*s[2:end]'/norm(s[2:end])*sigma) 
        T2 = sign(t[1])*sqrt(1-ϵ^2)*I+im*(ϵ*t[2:end]'/norm(t[2:end])*sigma)

        R,S,T = su3_from_su2(R2,S2,T2)
        X = R*S*T
        if rand(rng) < 0.5
            return X
        else 
            return X'
        end
    end

    function su3_from_su2(R2,S2,T2)
        R = SMatrix{3,3}([R2[1,1] R2[1,2] 0 ; R2[2,1] R2[2,2] 0 ; 0 0 1])
        S = SMatrix{3,3}([S2[1,1] 0 S2[1,2] ; 0 1 0 ; S2[2,1] 0 S2[2,2]])
        T = SMatrix{3,3}([1 0 0 ; 0 T2[1,1] T2[1,2] ; 0 T2[2,1] T2[2,2]])
        return R,S,T
    end

end

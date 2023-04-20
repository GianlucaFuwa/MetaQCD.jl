module Utils
    using LinearAlgebra
    using Random
    using StaticArrays

    export exp_iQ,
    gen_SU3_matrix,
    proj_onto_SU3,
    KenneyLaub,
    SU2_from_SU3,
    SU3_from_SU2,
    is_SU3,is_su3,
    trAB,
    calc_coefficients_Q,
    Traceless_antihermitian,Traceless_hermitian,Antihermitian,Hermitian

    function trAB(A;B=A)
        trAB = 0.0
        for i = 1:3
            for j = 1:3
                trAB += A[i,j]*B[j,i]
            end
        end
        return trAB
    end

    # Exponential function of iQ ∈ 𝔰𝔲(3), i.e. Projection of Q onto SU(3)
    # From Morningstar & Peardon (2008) arXiv:hep-lat/0311018v1
    function exp_iQ(Q::SMatrix{3,3,ComplexF64,9})
        if norm(Q)>1e-10
            u,w = set_uw(Q)
            f0,f1,f2,_ = set_fj(u,w)
            expiQ = f0*I + f1*Q + f2*Q^2
            return expiQ
        else
            expiQ = SMatrix{3,3,ComplexF64}(I)
            return expiQ

        end
    end

    function set_uw(Q::SMatrix{3,3,ComplexF64,9})
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
        return f0,f1,f2,ξ0
    end

    function calc_coefficients_Q(Q::SMatrix{3,3,ComplexF64,9})
        u,w = set_uw(Q)
        f0,f1,f2,ξ0 = set_fj(u,w)
        e2iu = exp(2im*u)
        emiu = exp(-im*u)
        cosw = cos(w)
        ξ1 = cosw/w^2 - sin(w)/w^3

        r10 = 2*( u + im*(u^2-w^2) )*e2iu +
            2emiu*( 4u*(2-im*u)*cosw + im*(9u^2+w^2-im*u*(3u^2+w^2))*ξ0 )
        r11 = 2*(1+2im*u)*e2iu + emiu*( -2*(1-im*u)*cosw + im*(6u+im*(w^2-3u^2))*ξ0 )
        r12 = 2im*e2iu + im*emiu*( cosw - 3*(1-im*u)*ξ0 )
        r20 = -2*e2iu + 2im*u*emiu*( cosw + (1+4im*u)*ξ0 +3u^2*ξ1 )
        r21 = -im*emiu*( cosw + (1+2im*u)*ξ0 - 3u^2*ξ1 )
        r22 = emiu*( ξ0 -3im*u*ξ1 )

        bdenom = 2*(9u^2-w^2)^2
        b10 = ( 2u*r10 + (3u^2-w^2)*r20 - 2*(15u^2+w^2)*f0 ) / bdenom
        b20 = ( r10 - 3u*r20 - 24u*f0 ) / bdenom
        b11 = ( 2u*r11 + (3u^2-w^2)*r21 - 2*(15u^2+w^2)*f1 ) / bdenom
        b21 = ( r11 - 3u*r21 - 24u*f1 ) / bdenom
        b12 = ( 2u*r12 + (3u^2-w^2)*r22 - 2*(15u^2+w^2)*f2 ) / bdenom
        b22 = ( r12 - 3u*r22 - 24u*f2 ) / bdenom

        return f0, f1, f2, b10, b11, b12, b20, b21, b22
    end

    const sigma1 = SMatrix{2,2,ComplexF64}([
        0 1
        1 0
    ])
    const sigma2 = SMatrix{2,2,ComplexF64}([
        0  -im
        im  0
    ])
    const sigma3 = SMatrix{2,2,ComplexF64}([
        1  0
        0 -1
    ])
    const pauli_vec = [sigma1, sigma2, sigma3]
    const eye2 = SMatrix{2,2}([
        1 0
        0 1
    ])
    const eye3 = SMatrix{3,3}([
        1 0 0
        0 1 0
        0 0 1
    ])


    # Generator of local Update Proposal Matrices X
    # From Gattringer C. & Lang C.B. (Springer, Berlin Heidelberg 2010)
    function gen_SU3_matrix(rng::Xoshiro, ϵ::Float64)
        R2 = gen_SU2_matrix(rng, ϵ)
        S2 = gen_SU2_matrix(rng, ϵ)
        T2 = gen_SU2_matrix(rng, ϵ)

        R = SU3_from_SU2(R2, 1)
        S = SU3_from_SU2(S2, 2)
        T = SU3_from_SU2(T2, 3)
        X = R * S * T

        if rand(rng) < 0.5
            return X
        else
            return X'
        end
    end

    function gen_SU2_matrix(rng::Xoshiro, ϵ::Float64)
        r1 = rand(rng) - 0.5
        r2 = rand(rng) - 0.5
        r3 = rand(rng) - 0.5
        rnorm = sqrt(r1^2 + r2^2 + r3^2)
        X = sqrt(1-ϵ^2) * eye2 + im*ϵ/rnorm*(r1*sigma1 + r2*sigma2 + r3*sigma3)
        return X
    end

    function SU3_from_SU2(M::SMatrix{2,2,ComplexF64,4}, i)
        R = zeros(ComplexF64, 3, 3)
        if i == 1
            R[1,1] = M[1,1]
            R[1,2] = M[1,2]
            R[2,1] = M[2,1]
            R[2,2] = M[2,2]
            R[3,3] = 1.0
        elseif i == 2
            R[1,1] = M[1,1]
            R[1,3] = M[1,2]
            R[3,1] = M[2,1]
            R[3,3] = M[2,2]
            R[2,2] = 1.0
        elseif i == 3
            R[2,2] = M[1,1]
            R[2,3] = M[1,2]
            R[3,2] = M[2,1]
            R[3,3] = M[2,2]
            R[1,1] = 1.0
        end
        return SMatrix{3,3,ComplexF64}(R)
    end

    function SU2_from_SU3(M::SMatrix{3,3,ComplexF64,9}, i)
        if i == 1
            r11 = M[1,1]
            r12 = M[1,2]
            r21 = M[2,1]
            r22 = M[2,2]
            R = SMatrix{2,2,ComplexF64}([
                r11 r21
                r21 r22
                ])
            return R
        elseif i == 2
            s11 = M[1,1]
            s12 = M[1,3]
            s21 = M[3,1]
            s22 = M[3,3]
            S = SMatrix{2,2,ComplexF64}([
                s11 s21
                s21 s22
                ])
            return S
        elseif i == 3
            t11 = M[2,2]
            t12 = M[2,3]
            t21 = M[3,2]
            t22 = M[3,3]
            T = SMatrix{2,2,ComplexF64}([
                t11 t21
                t21 t22
                ])
            return T
        end
    end

    function proj_onto_SU2(M)
        a = M[1,1]*0.5 + conj(M[2,2])*0.5
        b = M[2,1]*0.5 + conj(M[1,2])*0.5
        S = SMatrix{2,2}([a -conj(b); b conj(a)])
        return S
    end

    function proj_onto_SU2(M)
        a = M[1,1]*0.5 + conj(M[2,2])*0.5
        b = M[2,1]*0.5 + conj(M[1,2])*0.5
        M[1,1] = a
        M[2,1] = b
        M[1,2] = -conj(b)
        M[2,2] = conj(a)
        return M
    end

    function proj_onto_SU3(M::SMatrix{3,3,ComplexF64,9})
        col1 = M[:,1]
        col2 = M[:,2]
        col3 = M[:,3]
        col1 /= norm(col1)
        col2 -= (col1'*col2)*col1
        col2 /= norm(col2)
        col3 -= (col1'*col3)*col1 + (col2'*col3)*col2
        col3 /= norm(col3)
        M_SU3 = [col1 col2 col3]
        return M_SU3 / det(M_SU3)^(1/3)
    end

    function KenneyLaub(M::SMatrix{3,3,ComplexF64,9})
        while true
            X = M' * M
            M = M/3.0 * (I + 8/3 * inv(M'*M + 1/3*I))
            if norm(I - X) <= 1e-6
                break
            end
        end
        M = M / det(M)^(1/3)
    end

    function is_SU3(M; prec = 1e-6)
        is_SU3_upto_prec = norm(I - M*M') < prec && abs(1 - det(M)) < prec 
        return is_SU3_upto_prec
    end

    function is_su3(M; prec = 1e-6)
        is_su3_upto_prec = norm(M - M') < prec && abs(tr(M)) < prec
        return is_su3_upto_prec
    end

    function make_submatrix(UV,i,j)
        s11 = UV[i,i]
        s12 = UV[i,j]
        s21 = UV[j,i]
        s22 = UV[j,j]
        return SMatrix{2,2}([s11 s12; s21 s22])
    end

    function make_submatrix!(S,UV,i,j)
        s11 = UV[i,i]
        s12 = UV[i,j]
        s21 = UV[j,i]
        s22 = UV[j,j]
        S = SMatrix{2,2}([s11 s12; s21 s22])
        return nothing
    end

    function Traceless_antihermitian(M::SMatrix{3,3,ComplexF64,9})
        return 0.5*(M - M') - 1/6*tr(M - M')*I
    end

    function Antihermitian(M::SMatrix{3,3,ComplexF64,9})
        return 0.5*(M - M')
    end

    function Traceless_hermitian(M::SMatrix{3,3,ComplexF64,9})
        return 0.5*(M + M') - 1/6*tr(M + M')*I
    end

    function Hermitian(M::SMatrix{3,3,ComplexF64,9})
        return 0.5*(M + M')
    end

end

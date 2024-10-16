# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
    """
    gen_SU3_matrix(ϵ, ::Type{T}) where {T}

    Generate a Matrix X ∈ SU(3) with precision `T` near the identity with spread `ϵ`. \\
    From Gattringer C. & Lang C.B. (Springer, Berlin Heidelberg 2010)
    """
    @inline function gen_SU3_matrix(ϵ, ::Type{T}) where {T}
        R2 = gen_SU2_matrix(ϵ, T)
        S2 = gen_SU2_matrix(ϵ, T)
        T2 = gen_SU2_matrix(ϵ, T)

        R3 = embed_into_SU3_12(R2)
        S3 = embed_into_SU3_13(S2)
        T3 = embed_into_SU3_23(T2)
        out = cmatmul_ooo(R3, S3, T3)

        if rand(T) < T(0.5)
            return out
        else
            return out'
        end
    end

    """
    gen_SU2_matrix(ϵ, ::Type{T}) where {T}

    Generate a Matrix X ∈ SU(2) with precision `T` near the identity with spread `ϵ`. \\
    From Gattringer C. & Lang C.B. (Springer, Berlin Heidelberg 2010)
    """
    @inline function gen_SU2_matrix(ϵ, ::Type{T}) where {T}
        onehalf = T(0.5)
        r₁ = rand(T) - onehalf
        r₂ = rand(T) - onehalf
        r₃ = rand(T) - onehalf
        rnorm = sqrt(r₁^2 + r₂^2 + r₃^2)
        out = sqrt(one(T) - ϵ^2) * eye2(T) + im * ϵ / rnorm * (r₁ * σ₁ + r₂ * σ₂ + r₃ * σ₃)
        return out
    end

    """
    rand_SU3(::Type{T}) where {T}

    Generate a random Matrix X ∈ SU(3) with precision `T`. \\
    """
    @inline function rand_SU3(::Type{T}) where {T}
        out = @SMatrix rand(Complex{T}, 3, 3)
        out = proj_onto_SU3(out)
        return out
    end

    @inline function materialize_TA(h₁::T, h₂::T, h₃::T, h₄::T, h₅::T, h₆::T, h₇::T, h₈::T) where {T}
        sq3i = 1 / sqrt(T(3))
        out = @SMatrix [
            im*(h₃+h₈*sq3i)     h₂+im*h₁   h₅+im*h₄
            -h₂+im*h₁   im*(-h₃+h₈*sq3i)   h₇+im*h₆
            -h₅+im*h₄   -h₇+im*h₆   im*(-2*h₈*sq3i)
        ]
        return out
    end

    """
    gaussian_TA_mat(::Type{T}) where {T}

    Generate a normally distributed traceless anti-Hermitian 3x3 matrix with precision `T`.
    """
    @inline function gaussian_TA_mat(::Type{T}) where {T}
        onehalf = T(0.5)
        h₁ = onehalf * randn(T)
        h₂ = onehalf * randn(T)
        h₃ = onehalf * randn(T)
        h₄ = onehalf * randn(T)
        h₅ = onehalf * randn(T)
        h₆ = onehalf * randn(T)
        h₇ = onehalf * randn(T)
        h₈ = onehalf * randn(T)
        out = materialize_TA(h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈)
        return out
    end

    # function gaussian_su3_matrix(params)
    #     randn!(params)
    #     h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈ = 0.5params
    #     sq3 = sqrt(3)
    #     out = @SMatrix [
    #         im*(h₃+h₈/sq3) h₂+im*h₁        h₅+im*h₄
    #         -h₂+im*h₁      im*(-h₃+h₈/sq3) h₇+im*h₆
    #         -h₅+im*h₄      -h₇+im*h₆       im*(-2*h₈/sq3)
    #     ]
    #     return out
    # end

    @inline embed_into_SU3_12(M::SMatrix{2,2,Complex{T},4}) where {T} = @SMatrix [
        M[1, 1] M[1, 2] zero(T)
        M[2, 1] M[2, 2] zero(T)
        zero(T) zero(T) one(T)
    ]

    @inline embed_into_SU3_13(M::SMatrix{2,2,Complex{T},4}) where {T} = @SMatrix [
        M[1, 1] zero(T) M[1, 2]
        zero(T) one(T) zero(T)
        M[2, 1] zero(T) M[2, 2]
    ]

    @inline embed_into_SU3_23(M::SMatrix{2,2,Complex{T},4}) where {T} = @SMatrix [
        one(T) zero(T) zero(T)
        zero(T) M[1, 1] M[1, 2]
        zero(T) M[2, 1] M[2, 2]
    ]

    @inline function make_submatrix_12(M::SMatrix{3,3,Complex{T},9}) where {T}
        onehalf = T(0.5)
        α = onehalf * (M[1, 1] + conj(M[2, 2]))
        β = onehalf * (M[2, 1] - conj(M[1, 2]))
        out = @SMatrix [
            α -conj(β)
            β conj(α)
        ]
        return out
    end

    @inline function make_submatrix_13(M::SMatrix{3,3,Complex{T},9}) where {T}
        onehalf = T(0.5)
        α = onehalf * (M[1, 1] + conj(M[3, 3]))
        β = onehalf * (M[3, 1] - conj(M[1, 3]))
        out = @SMatrix [
            α -conj(β)
            β conj(α)
        ]
        return out
    end

    @inline function make_submatrix_23(M::SMatrix{3,3,Complex{T},9}) where {T}
        onehalf = T(0.5)
        α = onehalf * (M[2, 2] + conj(M[3, 3]))
        β = onehalf * (M[3, 2] - conj(M[2, 3]))
        out = @SMatrix [
            α -conj(β)
            β conj(α)
        ]
        return out
    end

    @inline function proj_onto_SU2(M::SMatrix{2,2,Complex{T},4}) where {T}
        onehalf = T(0.5)
        α = onehalf * (M[1, 1] + conj(M[2, 2]))
        β = onehalf * (M[1, 2] - conj(M[2, 1]))
        out = @SMatrix [
            α -conj(β)
            β conj(α)
        ]
        return out
    end

    @inline function proj_onto_SU3(M::SMatrix{3,3,Complex{T},9}) where {T}
        col1 = M[:, 1]
        col2 = M[:, 2]
        col3 = M[:, 3]
        col1 /= norm(col1)
        col2 -= (col1' * col2) * col1
        col2 /= norm(col2)
        col3 -= (col1' * col3) * col1 + (col2' * col3) * col2
        col3 /= norm(col3)
        out = [col1 col2 col3]
        out /= det(out)^(T(1 / 3))
        return out
    end

    @inline function restore_last_col(M::SMatrix{3,3,Complex{T},9}) where {T}
        tmp = MMatrix{3,3,Complex{T},9}(M)
        tmp[1, 3] = conj(tmp[2, 1] * tmp[3, 2] - tmp[3, 1] * tmp[2, 2])
        tmp[2, 3] = conj(tmp[3, 1] * tmp[1, 2] - tmp[1, 1] * tmp[3, 2])
        tmp[3, 3] = conj(tmp[1, 1] * tmp[2, 2] - tmp[2, 1] * tmp[1, 2])
        return SMatrix{3,3,Complex{T},9}(tmp)
    end

    @inline function restore_last_row(M::SMatrix{3,3,Complex{T},9}) where {T}
        tmp = MMatrix{3,3,Complex{T},9}(M)
        tmp[3, 1] = conj(tmp[1, 2] * tmp[2, 3] - tmp[1, 3] * tmp[2, 2])
        tmp[3, 2] = conj(tmp[1, 3] * tmp[2, 1] - tmp[1, 1] * tmp[2, 3])
        tmp[3, 3] = conj(tmp[1, 1] * tmp[2, 2] - tmp[1, 2] * tmp[2, 1])
        return SMatrix{3,3,Complex{T},9}(tmp)
    end

    """
    kenney_laub(M::SMatrix{3,3,Complex{T},9}) where {T}

    Compute the SU(3) matrix closest to `M` using the Kenney-Laub algorithm.
    """
    function kenney_laub(M::SMatrix{3,3,Complex{T},9}) where {T}
        onethird = T(1 / 3)
        eightthirds = T(8 / 3)
        for _ in 1:25
            X = cmatmul_do(M, M)
            norm(eye3(T) - X) <= convert(T, 1e-12) && break
            M = cmatmul_oo(
                onethird * M, (eye3(T) + eightthirds * inv(X + onethird * eye3(T)))
            )
            # M_enum = 5eye3 + 10X + cmatmul_oo(X, X)
            # M_denom = inv(eye3 + 10X + 5cmatmul_oo(X, X))
            # M = cmatmul_oo(M, cmatmul_oo(M_enum, M_denom))
        end

        M /= det(M)^(T(1 / 3))
        return M
    end

    @inline function is_special_unitary(M::SMatrix{3,3,Complex{T},9}, prec=1e-12) where {T}
        is_SU3 = norm(eye3(T) - cmatmul_od(M, M)) < prec && abs(one(T) - det(M)) < prec
        return is_SU3
    end

    @inline function is_traceless_antihermitian(
        M::SMatrix{N,N,Complex{T},N²}, prec=1e-12
    ) where {T,N,N²}
        is_TA = norm(M + M') < prec && abs(tr(M)) < prec
        return is_TA
    end

    @inline function is_antihermitian(
        M::SMatrix{N,N,Complex{T},N²}, prec=1e-12
    ) where {T,N,N²}
        is_TA = norm(M + M') < prec
        return is_TA
    end

    @inline function is_traceless_hermitian(
        M::SMatrix{N,N,Complex{T},N²}, prec=1e-12
    ) where {T,N,N²}
        is_TH = norm(M - M') < prec && abs(tr(M)) < prec
        return is_TH
    end

    @inline function is_hermitian(
        M::SMatrix{N,N,Complex{T},N²}, prec=1e-12
    ) where {T,N,N²}
        is_TH = norm(M - M') < prec
        return is_TH
    end

    @inline function traceless_antihermitian(M::SMatrix{N,N,Complex{T},N²}) where {T,N,N²}
        out = T(0.5) * (M - M') - T(1 / 2N) * tr(M - M') * one(M)
        return out
    end

    @inline function antihermitian(M::SMatrix{N,N,Complex{T},N²}) where {T,N,N²}
        out = T(0.5) * (M - M')
        return out
    end

    @inline function traceless_hermitian(M::SMatrix{N,N,Complex{T},N²}) where {T,N,N²}
        out = T(0.5) * (M + M') - T(1 / 2N) * tr(M + M') * one(M)
        return out
    end

    @inline function hermitian(M::SMatrix{N,N,Complex{T},N²}) where {T,N,N²}
        out = T(0.5) * (M + M')
        return out
    end
end

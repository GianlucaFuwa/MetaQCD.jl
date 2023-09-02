"""
Generator of Matrices X ∈ SU(3) near the identity \\
From Gattringer C. & Lang C.B. (Springer, Berlin Heidelberg 2010)
"""
function gen_SU3_matrix(ϵ)
    R2 = gen_SU2_matrix(ϵ)
    S2 = gen_SU2_matrix(ϵ)
    T2 = gen_SU2_matrix(ϵ)

    R = embed_into_SU3(R2, 1, 2)
    S = embed_into_SU3(S2, 1, 3)
    T = embed_into_SU3(T2, 2, 3)
    out = cmatmul_ooo(R, S, T)

    if rand() < 0.5
        return out
    else
        return out'
    end
end

function gen_SU2_matrix(ϵ)
    r₁ = rand() - 0.5
    r₂ = rand() - 0.5
    r₃ = rand() - 0.5
    rnorm = sqrt(r₁^2 + r₂^2 + r₃^2)
    out = sqrt(1 - ϵ^2) * eye2 + im * ϵ/rnorm * (r₁ * σ₁ + r₂ * σ₂ + r₃ * σ₃)
    return out
end

function gaussian_su3_matrix()
    sq3 = sqrt(3)
    h₁ = 0.5 * randn(Float64)
    h₂ = 0.5 * randn(Float64)
    h₃ = 0.5 * randn(Float64)
    h₄ = 0.5 * randn(Float64)
    h₅ = 0.5 * randn(Float64)
    h₆ = 0.5 * randn(Float64)
    h₇ = 0.5 * randn(Float64)
    h₈ = 0.5 * randn(Float64)
    out = @SMatrix [
        im*(h₃+h₈/sq3) h₂+im*h₁        h₅+im*h₄
        -h₂+im*h₁      im*(-h₃+h₈/sq3) h₇+im*h₆
        -h₅+im*h₄      -h₇+im*h₆       im*(-2*h₈/sq3)
    ]
    return out
end

function gaussian_su3_matrix(h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈)
    sq3 = sqrt(3)
    out = @SMatrix [
        0.5im*(h₃+h₈/sq3) 0.5h₂+0.5im*h₁     0.5h₅+0.5im*h₄
        -0.5h₂+0.5im*h₁   0.5im*(-h₃+h₈/sq3) 0.5h₇+0.5im*h₆
        -0.5h₅+0.5im*h₄   -0.5h₇+0.5im*h₆    0.5im*(-2*h₈/sq3)
    ]
    return out
end

function embed_into_SU3(M::SMatrix{2,2,ComplexF64,4}, i, j)
    if (i, j) == (1, 2)
        out = @SMatrix [
            M[1,1] M[1,2] 0
            M[2,1] M[2,2] 0
            0.0000 0.0000 1
        ]
    elseif (i, j) == (2, 3)
        out = @SMatrix [
            1 0.0000 0.0000
            0 M[1,1] M[1,2]
            0 M[2,1] M[2,2]
        ]
    elseif (i, j) == (1, 3)
        out = @SMatrix [
            M[1,1] 0 M[1,2]
            0.0000 1 0.0000
            M[2,1] 0 M[2,2]
        ]
    else
        out = eye3
    end

    return out
end

function make_submatrix(M::SMatrix{3,3,ComplexF64,9}, i, j)
    α = 0.5 * (M[i,i] + conj(M[j,j]))
    β = 0.5 * (M[j,i] - conj(M[i,j]))
    out = @SMatrix [
        α -conj(β)
        β  conj(α)
    ]
    return out
end

function proj_onto_SU2(M::SMatrix{2,2,ComplexF64,4})
    α = 0.5 * (M[1,1] + conj(M[2,2]))
    β = 0.5 * (M[1,2] - conj(M[2,1]))
    out = @SMatrix [
        α -conj(β)
        β  conj(α)
    ]
    return out
end

function proj_onto_SU3(M::SMatrix{3,3,ComplexF64,9})
    col1 = M[:,1]
    col2 = M[:,2]
    col3 = M[:,3]
    col1 /= norm(col1)
    col2 -= (col1' * col2) * col1
    col2 /= norm(col2)
    col3 -= (col1' * col3) * col1 + (col2' * col3) * col2
    col3 /= norm(col3)
    out = [col1 col2 col3]
    out /= det(out)^(1/3)
    return out
end

function kenney_laub(M::SMatrix{3,3,ComplexF64,9})
    for _ in 1:25
        X = cmatmul_do(M, M)
        norm(eye3 - X) <= 1e-12 && break
        M = cmatmul_oo(M/3, (eye3 + 8/3 * inv(X + 1/3 * eye3)))
        # M_enum = 5eye3 + 10X + cmatmul_oo(X, X)
        # M_denom = inv(eye3 + 10X + 5cmatmul_oo(X, X))
        # M = cmatmul_oo(M, cmatmul_oo(M_enum, M_denom))
    end

    M /= det(M)^(1/3)
    return M
end

function is_special_unitary(M::SMatrix{3,3,ComplexF64,9}, prec = 1e-12)
    is_SU3_upto_prec = norm(eye3 - cmatmul_od(M, M)) < prec && abs(1 - det(M)) < prec
    return is_SU3_upto_prec
end

function is_traceless_antihermitian(M::SMatrix{3,3,ComplexF64,9}, prec = 1e-12)
    is_su3_upto_prec = norm(M - M') < prec && abs(tr(M)) < prec
    return is_su3_upto_prec
end

function traceless_antihermitian(M::SMatrix{3,3,ComplexF64,9})
    out = 0.5 * (M - M') - 1/6 * tr(M - M') * eye3
    return out
end

function antihermitian(M::SMatrix{3,3,ComplexF64,9})
    out = 0.5 * (M - M')
    return out
end

function traceless_hermitian(M::SMatrix{3,3,ComplexF64,9})
    out = 0.5 * (M + M') - 1/6 * tr(M + M') * eye3
    return out
end

function hermitian(M::SMatrix{3,3,ComplexF64,9})
    out = 0.5 * (M + M')
    return out
end

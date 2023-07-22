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
    r1 = rand() - 0.5
    r2 = rand() - 0.5
    r3 = rand() - 0.5
    rnorm = sqrt(r1^2 + r2^2 + r3^2)
    out = sqrt(1 - ϵ^2) * eye2 +
        im * ϵ/rnorm * (r1 * σ1 + r2 * σ2 + r3 * σ3)
    return out
end

function gaussian_su3_matrix()
    sq3 = sqrt(3)
    h1 = 0.5 * randn(Float64)
    h2 = 0.5 * randn(Float64)
    h3 = 0.5 * randn(Float64)
    h4 = 0.5 * randn(Float64)
    h5 = 0.5 * randn(Float64)
    h6 = 0.5 * randn(Float64)
    h7 = 0.5 * randn(Float64)
    h8 = 0.5 * randn(Float64)
    out = @SMatrix [
        im*(h3+h8/sq3) h2+im*h1        h5+im*h4
        -h2+im*h1      im*(-h3+h8/sq3) h7+im*h6
        -h5+im*h4      -h7+im*h6       im*(-2*h8/sq3)
    ]
    return out
end

function gaussian_su3_matrix(h1, h2, h3, h4, h5, h6, h7, h8)
    sq3 = sqrt(3)
    out = @SMatrix [
        0.5im*(h3+h8/sq3) 0.5h2+0.5im*h1     0.5h5+0.5im*h4
        -0.5h2+0.5im*h1   0.5im*(-h3+h8/sq3) 0.5h7+0.5im*h6
        -0.5h5+0.5im*h4   -0.5h7+0.5im*h6    0.5im*(-2*h8/sq3)
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
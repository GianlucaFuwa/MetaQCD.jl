"""
Generator of Matrices X ∈ SU(3) near the identity \\
From Gattringer C. & Lang C.B. (Springer, Berlin Heidelberg 2010)
"""
function gen_SU3_matrix(rng, ϵ)
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

function gen_SU2_matrix(rng, ϵ)
    r1 = rand(rng) - 0.5
    r2 = rand(rng) - 0.5
    r3 = rand(rng) - 0.5
    rnorm = sqrt(r1^2 + r2^2 + r3^2)
    X = sqrt(1 - ϵ^2) * eye2 + 
        im * ϵ/rnorm * (r1 * σ1 + r2 * σ2 + r3 * σ3)
    return X
end

function embed_into_SU3(M::SMatrix{2,2,T,4}, i, j) where {T}
    M_SU3 = zeros(ComplexF64, 3, 3)

    for n = 1:3
        M_SU3[n,n] = 1
    end

    M_SU3[i,i] = M[1,1]
    M_SU3[i,j] = M[1,2]
    M_SU3[j,i] = M[2,1]
    M_SU3[j,j] = M[2,2]
    return SMatrix{3,3,ComplexF64}(M_SU3)
end

function make_submatrix(M::SMatrix{3,3,ComplexF64,9}, i, j)
    α = 0.5 * (M[i,i] + conj(M[j,j]))
    β = 0.5 * (M[j,i] - conj(M[i,j]))
    M_SU2 = @SMatrix [
        α -conj(β)
        β  conj(α)
    ]
    return M_SU2
end

function proj_onto_SU2(M::SMatrix{2,2,ComplexF64,4})
    α = 0.5 * (M[1,1] + conj(M[2,2]))
    β = 0.5 * (M[1,2] - conj(M[2,1]))
    S = @SMatrix [
        α -conj(β)
        β  conj(α)
    ]
    return S
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
    M_SU3 = [col1 col2 col3]
    return M_SU3 / det(M_SU3)^(1/3)
end

function kenney_laub(M::SMatrix{3,3,ComplexF64,9})
    while true
        X = M' * M

        if norm(eye3 - X) <= 1e-6
            break
        end

        M = M / 3 * (eye3 + 8/3 * inv(M' * M + 1/3 * eye3))
    end

    M = M / det(M)^(1/3)
    return M
end

function is_special_unitary(M::SMatrix{3,3,ComplexF64,9}, prec = 1e-6)
    is_SU3_upto_prec = norm(eye3 - M * M') < prec && abs(1 - det(M)) < prec 
    return is_SU3_upto_prec
end

function is_traceless_antihermitian(M::SMatrix{3,3,ComplexF64,9}, prec = 1e-6)
    is_su3_upto_prec = norm(M - M') < prec && abs(tr(M)) < prec
    return is_su3_upto_prec
end

function traceless_antihermitian(M::SMatrix{3,3,ComplexF64,9})
    return 0.5 * (M - M') - 1/6 * tr(M - M') * eye3
end

function antihermitian(M::SMatrix{3,3,ComplexF64,9})
    return 0.5 * (M - M')
end

function traceless_hermitian(M::SMatrix{3,3,ComplexF64,9})
    return 0.5 * (M + M') - 1/6 * tr(M + M') * eye3
end

function hermitian(M::SMatrix{3,3,ComplexF64,9})
    return 0.5 * (M + M')
end
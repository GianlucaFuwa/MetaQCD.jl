const WilsonFermionfield{B,T,A} = Fermionfield{B,T,A,4}

function calc_dSfdU!(dU, fermion_action::WilsonFermionAction{2}, U, ϕ::WilsonFermionfield)
    @assert dims(dU) == dims(U) == dims(ϕ)
    clear!(dU)
    cg_tol = fermion_action.cg_tol
    cg_maxiters = fermion_action.cg_maxiters
    X, Y, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)

    clear!(X)
    solve_D⁻¹x!(X, DdagD, ϕ, Y, temp1, temp2, cg_tol, cg_maxiters) # Y is used here merely as a temp LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    add_wilson_derivative!(dU, U, X, Y, D.anti_periodic)
    D.csw != 0 && add_clover_derivative!(dU, U, X, Y, D.csw)
    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::WilsonFermionAction{Nf}, U, ϕ::WilsonFermionfield
) where {Nf}
    @assert dims(dU) == dims(U) == dims(ϕ)
    clear!(dU)
    cg_tol = fermion_action.cg_tol
    cg_maxiters = fermion_action.cg_maxiters
    rhmc = fermion_action.rhmc_info_md
    n = rhmc.coeffs.n
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    Xs = fermion_action.rhmc_temps1
    Ys = fermion_action.rhmc_temps2
    temp1, temp2 = fermion_action.cg_temps

    for v in Xs
        clear!(v)
    end
    shifts = rhmc.coeffs_inverse.β
    coeffs = rhmc.coeffs_inverse.α
    solve_D⁻¹x_multishift!(Xs, shifts, DdagD, ϕ, temp1, temp2, Ys, cg_tol, cg_maxiters)

    for i in 1:n
        LinearAlgebra.mul!(Ys[i+1], D, Xs[i+1]) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
        add_wilson_derivative!(dU, U, Xs[i+1], Ys[i+1], D.anti_periodic; coeff=coeffs[i])
        if D.csw != 0
            add_clover_derivative!(dU, U, Xs[i+1], Ys[i+1], D.csw; coeff=coeffs[i])
        end
    end
    return nothing
end

function add_wilson_derivative!(
    dU, U, X::T, Y::T, anti; coeff=1
) where {T<:WilsonFermionfield}
    @assert dims(dU) == dims(U) == dims(X) == dims(Y)
    NX, NY, NZ, NT = dims(U)
    fac = float_type(U)(0.5coeff)

    for site in eachindex(dU)
        siteμ⁺ = move(site, 1, 1, NX)
        B = ckron_sum(spin_proj(X[siteμ⁺], Val(-1)), Y[site])
        C = ckron_sum(spin_proj(Y[siteμ⁺], Val(1)), X[site])
        dU[1, site] += fac * traceless_antihermitian(cmatmul_oo(U[1, site], B + C))

        siteμ⁺ = move(site, 2, 1, NY)
        B = ckron_sum(spin_proj(X[siteμ⁺], Val(-2)), Y[site])
        C = ckron_sum(spin_proj(Y[siteμ⁺], Val(2)), X[site])
        dU[2, site] += fac * traceless_antihermitian(cmatmul_oo(U[2, site], B + C))

        siteμ⁺ = move(site, 3, 1, NZ)
        B = ckron_sum(spin_proj(X[siteμ⁺], Val(-3)), Y[site])
        C = ckron_sum(spin_proj(Y[siteμ⁺], Val(3)), X[site])
        dU[3, site] += fac * traceless_antihermitian(cmatmul_oo(U[3, site], B + C))

        siteμ⁺ = move(site, 4, 1, NT)
        bc⁺ = boundary_factor(anti, site[4], 1, NT)
        B = ckron_sum(spin_proj(X[siteμ⁺], Val(-4)), Y[site])
        C = ckron_sum(spin_proj(Y[siteμ⁺], Val(4)), X[site])
        dU[4, site] += bc⁺ * fac * traceless_antihermitian(cmatmul_oo(U[4, site], B + C))
    end
end

function add_clover_derivative!(
    dU, U, X::T, Y::T, csw; coeff=1
) where {T<:WilsonFermionfield}
    @assert dims(dU) == dims(U) == dims(X) == dims(Y)
    Tf = float_type(U)
    fac = Tf(-csw * coeff / 2)

    for site in eachindex(dU)
        tmp1 = ∇σμνFμν(U, site, Val(1), Tf)
        tmp2 = ckron_sum(cmvmul(tmp1, X[site]), Y[site])
        dU[1, site] += fac * traceless_antihermitian(cmatmul_oo(U[1, site], tmp2 + tmp2'))

        tmp1 = ∇σμνFμν(U, site, Val(2), Tf)
        tmp2 = ckron_sum(cmvmul(tmp1, X[site]), Y[site])
        dU[2, site] += fac * traceless_antihermitian(cmatmul_oo(U[2, site], tmp2 + tmp2'))

        tmp1 = ∇σμνFμν(U, site, Val(3), Tf)
        tmp2 = ckron_sum(cmvmul(tmp1, X[site]), Y[site])
        dU[3, site] += fac * traceless_antihermitian(cmatmul_oo(U[3, site], tmp2 + tmp2'))

        tmp1 = ∇σμνFμν(U, site, Val(4), Tf)
        tmp2 = ckron_sum(cmvmul(tmp1, X[site]), Y[site])
        dU[4, site] += fac * traceless_antihermitian(cmatmul_oo(U[4, site], tmp2 + tmp2'))
    end
end

function ∇σμνFμν(U, site, ::Val{1}, ::Type{T}) where {T}
    NX, NY, NZ, NT = dims(U)
    out = @SMatrix zeros(Complex{T}, 12, 12)
    siteμ⁺ = move(site, 1i32, 1i32, NX)

    siteν⁺ = move(site, 2i32, 1i32, NY)
    tmp = cmatmul_odd(U[2i32, siteμ⁺], U[1i32, siteν⁺], U[2i32, site])
    out += ckron(σ₁₂(T), tmp)

    siteν⁺ = move(site, 3i32, 1i32, NZ)
    tmp = cmatmul_odd(U[3i32, siteμ⁺], U[1i32, siteν⁺], U[3i32, site])
    out += ckron(σ₁₃(T), tmp)

    siteν⁺ = move(site, 4i32, 1i32, NT)
    tmp = cmatmul_odd(U[4i32, siteμ⁺], U[1i32, siteν⁺], U[4i32, site])
    out += ckron(σ₁₄(T), tmp)
    return Complex{T}(im / 4) * out
end

function ∇σμνFμν(U, site, ::Val{2}, ::Type{T}) where {T}
    NX, NY, NZ, NT = dims(U)
    out = @SMatrix zeros(Complex{T}, 12, 12)
    siteμ⁺ = move(site, 2i32, 1i32, NX)

    siteν⁺ = move(site, 1i32, 1i32, NY)
    tmp = cmatmul_odd(U[1i32, siteμ⁺], U[2i32, siteν⁺], U[1i32, site])
    out += ckron(σ₁₂(T), tmp)

    siteν⁺ = move(site, 3i32, 1i32, NZ)
    tmp = cmatmul_odd(U[3i32, siteμ⁺], U[2i32, siteν⁺], U[3i32, site])
    out += ckron(σ₂₃(T), tmp)

    siteν⁺ = move(site, 4i32, 1i32, NT)
    tmp = cmatmul_odd(U[4i32, siteμ⁺], U[2i32, siteν⁺], U[4i32, site])
    out += ckron(σ₂₄(T), tmp)
    return Complex{T}(im / 4) * out
end

function ∇σμνFμν(U, site, ::Val{3}, ::Type{T}) where {T}
    NX, NY, NZ, NT = dims(U)
    out = @SMatrix zeros(Complex{T}, 12, 12)
    siteμ⁺ = move(site, 3i32, 1i32, NX)

    siteν⁺ = move(site, 1i32, 1i32, NY)
    tmp = cmatmul_odd(U[1i32, siteμ⁺], U[3i32, siteν⁺], U[1i32, site])
    out += ckron(σ₁₃(T), tmp)

    siteν⁺ = move(site, 2i32, 1i32, NZ)
    tmp = cmatmul_odd(U[2i32, siteμ⁺], U[3i32, siteν⁺], U[2i32, site])
    out += ckron(σ₂₃(T), tmp)

    siteν⁺ = move(site, 4i32, 1i32, NT)
    tmp = cmatmul_odd(U[4i32, siteμ⁺], U[3i32, siteν⁺], U[4i32, site])
    out += ckron(σ₃₄(T), tmp)
    return Complex{T}(im / 4) * out
end

function ∇σμνFμν(U, site, ::Val{4}, ::Type{T}) where {T}
    NX, NY, NZ, NT = dims(U)
    out = @SMatrix zeros(Complex{T}, 12, 12)
    siteμ⁺ = move(site, 4i32, 1i32, NX)

    siteν⁺ = move(site, 1i32, 1i32, NY)
    tmp = cmatmul_odd(U[1i32, siteμ⁺], U[4i32, siteν⁺], U[1i32, site])
    out += ckron(σ₁₄(T), tmp)

    siteν⁺ = move(site, 2i32, 1i32, NZ)
    tmp = cmatmul_odd(U[2i32, siteμ⁺], U[4i32, siteν⁺], U[2i32, site])
    out += ckron(σ₂₄(T), tmp)

    siteν⁺ = move(site, 3i32, 1i32, NT)
    tmp = cmatmul_odd(U[3i32, siteμ⁺], U[4i32, siteν⁺], U[3i32, site])
    out += ckron(σ₃₄(T), tmp)
    return Complex{T}(im / 4) * out
end

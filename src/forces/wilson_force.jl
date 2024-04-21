const WilsonFermionfield{B,T,A} = Fermionfield{B,T,A,4}

function calc_dSfdU!(dU, fermion_action::WilsonFermionAction, U, ϕ::WilsonFermionfield)
    @assert dims(dU) == dims(U) == dims(ϕ)
    X = fermion_action.cg_x
    Y, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D
    replace_U!(D, U)
    DdagD = DdaggerD(D)
    clear!(X)
    solve_D⁻¹x!(X, DdagD, ϕ, Y, temp1, temp2) # Y is used here merely as a temp
    LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    wilson_derivative!(dU, U, X, Y, D.anti_periodic)
    if D.csw !== nothing
        add_clover_derivative!(dU, U, X, Y, D.csw)
    end
    return nothing
end

function wilson_derivative!(dU, U, X::T, Y::T, anti) where {T<:WilsonFermionfield}
    @assert dims(dU) == dims(U) == dims(X) == dims(Y)
    NX, NY, NZ, NT = dims(U)
    onehalf = float_type(U)(0.5)

    for site in eachindex(dU)
        siteμ⁺ = move(site, 1, 1, NX)
        B = ckron_sum(spin_proj(X[siteμ⁺], Val(-1)), Y[site])
        C = ckron_sum(spin_proj(Y[siteμ⁺], Val(1)), X[site])
        dU[1, site] = onehalf * traceless_antihermitian(cmatmul_oo(U[1, site], B + C))

        siteμ⁺ = move(site, 2, 1, NY)
        B = ckron_sum(spin_proj(X[siteμ⁺], Val(-2)), Y[site])
        C = ckron_sum(spin_proj(Y[siteμ⁺], Val(2)), X[site])
        dU[2, site] = onehalf * traceless_antihermitian(cmatmul_oo(U[2, site], B + C))

        siteμ⁺ = move(site, 3, 1, NZ)
        B = ckron_sum(spin_proj(X[siteμ⁺], Val(-3)), Y[site])
        C = ckron_sum(spin_proj(Y[siteμ⁺], Val(3)), X[site])
        dU[3, site] = onehalf * traceless_antihermitian(cmatmul_oo(U[3, site], B + C))

        siteμ⁺ = move(site, 4, 1, NT)
        bc⁺ = boundary_factor(anti, site[4], 1, NT)
        B = ckron_sum(spin_proj(X[siteμ⁺], Val(-4)), Y[site])
        C = ckron_sum(spin_proj(Y[siteμ⁺], Val(4)), X[site])
        dU[4, site] = bc⁺ * onehalf * traceless_antihermitian(cmatmul_oo(U[4, site], B + C))
    end
end

function add_clover_derivative!(dU, U, X::T, Y::T, csw) where {T<:WilsonFermionfield}
    @assert dims(dU) == dims(U) == dims(X) == dims(Y)
    Tf = float_type(U)
    fac = Tf(-csw / 2)

    for site in eachindex(dU)
        tmp1 = ∇σμνFμν(U, Val(1), Tf)
        tmp2 = ckron_sum(cmvmul_color(tmp1, X[site]), Y[site])
        dU[1, site] += fac * traceless_antihermitian(cmatmul_oo(U[1, site], tmp2 + tmp2'))

        tmp1 = ∇σμνFμν(U, Val(2), Tf)
        tmp2 = ckron_sum(cmvmul_color(tmp1, X[site]), Y[site])
        dU[2, site] += fac * traceless_antihermitian(cmatmul_oo(U[2, site], tmp2 + tmp2'))

        tmp1 = ∇σμνFμν(U, Val(3), Tf)
        tmp2 = ckron_sum(cmvmul_color(tmp1, X[site]), Y[site])
        dU[3, site] += fac * traceless_antihermitian(cmatmul_oo(U[3, site], tmp2 + tmp2'))

        tmp1 = ∇σμνFμν(U, Val(4), Tf)
        tmp2 = ckron_sum(cmvmul_color(tmp1, X[site]), Y[site])
        dU[4, site] += fac * traceless_antihermitian(cmatmul_oo(U[4, site], tmp2 + tmp2'))
    end
end

function ∇σμνFμν(U, ::Val{1}, ::Type{T}) where {T} end

function ∇σμνFμν(U, ::Val{2}, ::Type{T}) where {T} end

function ∇σμνFμν(U, ::Val{3}, ::Type{T}) where {T} end

function ∇σμνFμν(U, ::Val{4}, ::Type{T}) where {T} end

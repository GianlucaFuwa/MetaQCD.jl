const WilsonSpinorfield{B,T,M,A} = Spinorfield{B,T,M,A,4}

function calc_dSfdU!(dU, fermion_action::WilsonFermionAction{2}, U, ϕ::WilsonSpinorfield)
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    X, Y, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)

    clear!(X)
    solve_dirac!(X, DdagD, ϕ, Y, temp1, temp2, cg_tol, cg_maxiters) # Y is used here merely as a temp LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    add_wilson_derivative!(dU, U, X, Y, D.boundary_condition)

    if has_clover_term(D)
        Xμν = fermion_action.Xμν
        calc_Xμν_eachsite!(Xμν, X, Y)
        add_clover_derivative!(dU, U, Xμν, D.csw)
    end

    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::WilsonFermionAction{Nf}, U, ϕ::WilsonSpinorfield
) where {Nf}
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    rhmc = fermion_action.rhmc_info_md
    n = get_n(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    bc = D.boundary_condition
    Xs = fermion_action.rhmc_temps1
    Ys = fermion_action.rhmc_temps2
    temp1, temp2 = fermion_action.cg_temps

    for X in Xs
        clear!(X)
    end

    shifts = get_β_inverse(rhmc)
    coeffs = get_α_inverse(rhmc)
    solve_dirac_multishift!(Xs, shifts, DdagD, ϕ, temp1, temp2, Ys, cg_tol, cg_maxiters)

    for i in 1:n
        LinearAlgebra.mul!(Ys[i+1], D, Xs[i+1]) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
        add_wilson_derivative!(dU, U, Xs[i+1], Ys[i+1], bc; coeff=coeffs[i])
        
        if has_clover_term(D)
            Xμν = fermion_action.Xμν
            calc_Xμν_eachsite!(Xμν, Xs[i+1], Ys[i+1])
            add_clover_derivative!(dU, U, Xμν, D.csw; coeff=coeffs[i])
        end
    end

    return nothing
end

function add_wilson_derivative!(
    dU::Colorfield{CPU,T}, U::Gaugefield{CPU,T}, X::TF, Y::TF, bc; coeff=1
) where {T,TF<:WilsonSpinorfield{CPU,T}}
    check_dims(dU, U, X, Y)
    fac = T(0.5coeff)

    # If we write out the kernel and use @batch, the program crashes for some reason
    # Stems from "pload" from StrideArraysCore.jl but ONLY if we write it out AND overload
    # "object_and_preserve" (cant reproduce in MWE yet)
    # is fine, because writing it like this makes the GPU port easier
    @batch for site in eachindex(dU)
        add_wilson_derivative_kernel!(dU, U, X, Y, site, bc, fac)
    end

    update_halo!(dU)
    return nothing
end

function add_wilson_derivative_kernel!(dU, U, X, Y, site, bc, fac)
    NX, NY, NZ, NT = dims(U)
    siteμ⁺ = move(site, 1, 1, NX)
    B = spintrace(spin_proj(X[siteμ⁺], Val(-1)), Y[site])
    C = spintrace(spin_proj(Y[siteμ⁺], Val(1)), X[site])
    dU[1i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[1, site], B + C))

    siteμ⁺ = move(site, 2, 1, NY)
    B = spintrace(spin_proj(X[siteμ⁺], Val(-2)), Y[site])
    C = spintrace(spin_proj(Y[siteμ⁺], Val(2)), X[site])
    dU[2i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[2, site], B + C))

    siteμ⁺ = move(site, 3, 1, NZ)
    B = spintrace(spin_proj(X[siteμ⁺], Val(-3)), Y[site])
    C = spintrace(spin_proj(Y[siteμ⁺], Val(3)), X[site])
    dU[3i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[3, site], B + C))

    siteμ⁺ = move(site, 4, 1, NT)
    B = spintrace(spin_proj(apply_bc(X[siteμ⁺], bc, site, Val(1), NT), Val(-4)), Y[site])
    C = spintrace(spin_proj(apply_bc(Y[siteμ⁺], bc, site, Val(1), NT), Val(4)), X[site])
    dU[4i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[4, site], B + C))
    return nothing
end

function add_clover_derivative!(
    dU::Colorfield{CPU,T}, U::Gaugefield{CPU,T}, Xμν::Tensorfield{CPU,T}, csw; coeff=1
) where {T}
    check_dims(dU, U, Xμν)
    fac = T(csw * coeff / 2)

    @batch for site in eachindex(dU)
        add_clover_derivative_kernel!(dU, U, Xμν, site, fac, T)
    end

    update_halo!(dU)
    return nothing
end

function add_clover_derivative_kernel!(dU, U, Xμν, site, fac, ::Type{T}) where {T}
    tmp =
        Xμν∇Fμν(Xμν, U, 1, 2, site, T) +
        Xμν∇Fμν(Xμν, U, 1, 3, site, T) +
        Xμν∇Fμν(Xμν, U, 1, 4, site, T)
    dU[1i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[1i32, site], tmp))

    tmp =
        Xμν∇Fμν(Xμν, U, 2, 1, site, T) +
        Xμν∇Fμν(Xμν, U, 2, 3, site, T) +
        Xμν∇Fμν(Xμν, U, 2, 4, site, T)
    dU[2i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[2i32, site], tmp))

    tmp =
        Xμν∇Fμν(Xμν, U, 3, 1, site, T) +
        Xμν∇Fμν(Xμν, U, 3, 2, site, T) +
        Xμν∇Fμν(Xμν, U, 3, 4, site, T)
    dU[3i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[3i32, site], tmp))

    tmp =
        Xμν∇Fμν(Xμν, U, 4, 1, site, T) +
        Xμν∇Fμν(Xμν, U, 4, 2, site, T) +
        Xμν∇Fμν(Xμν, U, 4, 3, site, T)
    dU[4i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[4i32, site], tmp))
    return nothing
end

function calc_Xμν_eachsite!(
    Xμν::Tensorfield{CPU,T}, X::TF, Y::TF
) where {T,TF<:WilsonSpinorfield}
    check_dims(Xμν, X, Y)

    @batch for site in eachindex(Xμν)
        calc_Xμν_kernel!(Xμν, X, Y, site)
    end

    return nothing
end

function calc_Xμν_kernel!(Xμν, X, Y, site)
    X₁₂ =
        spintrace(σμν_spin_mul(X[site], Val(1), Val(2)), Y[site]) +
        spintrace(σμν_spin_mul(Y[site], Val(1), Val(2)), X[site])
    Xμν[1i32, 2i32, site] = X₁₂
    Xμν[2i32, 1i32, site] = -X₁₂

    X₁₃ =
        spintrace(σμν_spin_mul(X[site], Val(1), Val(3)), Y[site]) +
        spintrace(σμν_spin_mul(Y[site], Val(1), Val(3)), X[site])
    Xμν[1i32, 3i32, site] = X₁₃
    Xμν[3i32, 1i32, site] = -X₁₃

    X₁₄ =
        spintrace(σμν_spin_mul(X[site], Val(1), Val(4)), Y[site]) +
        spintrace(σμν_spin_mul(Y[site], Val(1), Val(4)), X[site])
    Xμν[1i32, 4i32, site] = X₁₄
    Xμν[4i32, 1i32, site] = -X₁₄

    X₂₃ =
        spintrace(σμν_spin_mul(X[site], Val(2), Val(3)), Y[site]) +
        spintrace(σμν_spin_mul(Y[site], Val(2), Val(3)), X[site])
    Xμν[2i32, 3i32, site] = X₂₃
    Xμν[3i32, 2i32, site] = -X₂₃

    X₂₄ =
        spintrace(σμν_spin_mul(X[site], Val(2), Val(4)), Y[site]) +
        spintrace(σμν_spin_mul(Y[site], Val(2), Val(4)), X[site])
    Xμν[2i32, 4i32, site] = X₂₄
    Xμν[4i32, 2i32, site] = -X₂₄

    X₃₄ =
        spintrace(σμν_spin_mul(X[site], Val(3), Val(4)), Y[site]) +
        spintrace(σμν_spin_mul(Y[site], Val(3), Val(4)), X[site])
    Xμν[3i32, 4i32, site] = X₃₄
    Xμν[4i32, 3i32, site] = -X₃₄
    return nothing
end

function Xμν∇Fμν(Xμν, U, μ, ν, site, ::Type{T}) where {T}
    Nμ = dims(U)[μ]
    Nν = dims(U)[ν]
    siteμ⁺ = move(site, μ, 1i32, Nμ)
    siteν⁺ = move(site, ν, 1i32, Nν)
    siteν⁻ = move(site, ν, -1i32, Nν)
    siteμ⁺ν⁺ = move(siteμ⁺, ν, 1i32, Nν)
    siteμ⁺ν⁻ = move(siteμ⁺, ν, -1i32, Nν)

    # get reused matrices up to cache (can precalculate some products too)
    # Uνsiteμ⁺ = U[ν,siteμ⁺]
    # Uμsiteν⁺ = U[μ,siteν⁺]
    # Uνsite = U[ν,site]
    # Uνsiteμ⁺ν⁻ = U[ν,siteμ⁺ν⁻]
    # Uμsiteν⁻ = U[μ,siteν⁻]
    # Uνsiteν⁻ = U[ν,siteν⁻]

    component =
        cmatmul_oddo(U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site], Xμν[μ, ν, site]) +
        cmatmul_odod(U[ν, siteμ⁺], U[μ, siteν⁺], Xμν[μ, ν, siteν⁺], U[ν, site]) +
        cmatmul_oodd(U[ν, siteμ⁺], Xμν[μ, ν, siteμ⁺ν⁺], U[μ, siteν⁺], U[ν, site]) +
        cmatmul_oodd(Xμν[μ, ν, siteμ⁺], U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site]) -
        cmatmul_ddoo(U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻], Xμν[μ, ν, site]) -
        cmatmul_ddoo(U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], Xμν[μ, ν, siteν⁻], U[ν, siteν⁻]) -
        cmatmul_dodo(U[ν, siteμ⁺ν⁻], Xμν[μ, ν, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻]) -
        cmatmul_oddo(Xμν[μ, ν, siteμ⁺], U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻])

    return im * T(1/8) * component
end

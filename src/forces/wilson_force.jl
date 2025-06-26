function calc_dSfdU!(
    dU, fermion_action::FermionAction{false,2,TD}, U, ϕ::WilsonSpinorfield
) where {TD<:WilsonDiracOperator}
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    X, Y, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)

    clear!(X)
    iters, res = solve_dirac!(X, DdagD, ϕ, Y, temp1, temp2, cg_tol, cg_maxiters) # Y is used here merely as a temp LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!

    cg_datafile = fermion_action.cg_datafile

    if cg_datafile != ""
        set_ext!(cg_datafile, MPI_INSTANCE[])
        fp = fopen(cg_datafile, "a")
        printf(fp, "%-11i", iters)
        printf(fp, "%-25.15E", res)
        printf(fp, "%s", "# force")
        newline(fp)
        fclose(fp)
    end

    LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    add_wilson_derivative!(dU, U, X, Y, D.boundary_condition)

    if has_clover_term(D)
        Xμν = fermion_action.Xμν
        calc_Xμν_wilson_eachsite!(Xμν, X, Y)
        add_clover_derivative!(dU, U, Xμν, D.csw)
    end

    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::FermionAction{true,1,TD}, U, ϕ::WilsonSpinorfield
) where {TD<:WilsonDiracOperator}
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    rhmc = fermion_action.rhmc_info_md
    n = get_n_inverse(rhmc)
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
    iters, res = solve_dirac_multishift!(
        Xs, shifts, DdagD, ϕ, temp1, temp2, Ys, cg_tol, cg_maxiters
    )

    cg_datafile = fermion_action.cg_datafile

    if cg_datafile != ""
        set_ext!(cg_datafile, MPI_INSTANCE[])
        fp = fopen(cg_datafile, "a")
        printf(fp, "%-11i", iters)
        printf(fp, "%-25.15E", res)
        printf(fp, "%s", "# force")
        newline(fp)
        fclose(fp)
    end

    for i in 1:n
        LinearAlgebra.mul!(Ys[i+1], D, Xs[i+1]) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
        add_wilson_derivative!(dU, U, Xs[i+1], Ys[i+1], bc; coeff=coeffs[i])

        if has_clover_term(D)
            Xμν = fermion_action.Xμν
            calc_Xμν_wilson_eachsite!(Xμν, Xs[i+1], Ys[i+1])
            add_clover_derivative!(dU, U, Xμν, D.csw; coeff=coeffs[i])
        end
    end

    return nothing
end

function add_wilson_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T}, X::TF, Y::TF, bc; coeff=1
) where {B,T,TF<:WilsonSpinorfield{B,T}}
    fac = T(0.5coeff)
    # TODO: can hide
    update_halo!(U, X, Y)

    parallelfor(eachindex(dU, U, X, Y), B) do site
        add_wilson_derivative_kernel!(dU, U, X, Y, site, bc, fac)
    end

    return nothing
end

function add_wilson_derivative_kernel!(dU, U, X, Y, site, bc, fac)
    NT = size(dU, 4)

    siteμ⁺ = move(site, 1, 1, axes(dU, 1))
    B = spintrace(spin_proj(X[siteμ⁺], Val(-1)), Y[site])
    C = spintrace(spin_proj(Y[siteμ⁺], Val(1)), X[site])
    dU[1i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[1, site], B + C))

    siteμ⁺ = move(site, 2, 1, axes(dU, 2))
    B = spintrace(spin_proj(X[siteμ⁺], Val(-2)), Y[site])
    C = spintrace(spin_proj(Y[siteμ⁺], Val(2)), X[site])
    dU[2i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[2, site], B + C))

    siteμ⁺ = move(site, 3, 1, axes(dU, 3))
    B = spintrace(spin_proj(X[siteμ⁺], Val(-3)), Y[site])
    C = spintrace(spin_proj(Y[siteμ⁺], Val(3)), X[site])
    dU[3i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[3, site], B + C))

    siteμ⁺ = move(site, 4, 1, axes(dU, 4))
    B = spintrace(spin_proj(apply_bc(X[siteμ⁺], bc, site, Val(1), NT), Val(-4)), Y[site])
    C = spintrace(spin_proj(apply_bc(Y[siteμ⁺], bc, site, Val(1), NT), Val(4)), X[site])
    dU[4i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[4, site], B + C))
    return nothing
end

function add_clover_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T}, Xμν::Tensorfield{B,T}, csw; coeff=1
) where {B,T}
    fac = T(csw * coeff / 2)
    # INFO: we must have already updated the halo for the wilson derivative
    update_halo!(Xμν)

    parallelfor(eachindex(dU, U, Xμν), B) do site
        add_clover_derivative_kernel!(dU, U, Xμν, site, fac, T)
    end

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

function calc_Xμν_wilson_eachsite!(
    Xμν::Tensorfield{B,T}, X::TF, Y::TF
) where {B,T,TF<:WilsonSpinorfield{B}}
    parallelfor(eachindex(Xμν, X, Y), B) do site
        calc_Xμν_wilson_kernel!(Xμν, X, Y, site)
    end

    return nothing
end

function calc_Xμν_wilson_kernel!(Xμν, X, Y, site)
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
    Nμ = axes(U, μ)
    Nν = axes(U, ν)
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

    return im * T(1 / 8) * component
end

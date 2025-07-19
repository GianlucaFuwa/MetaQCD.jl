function calc_dSfdU!(
    dU, fermion_action::FermionAction{false,2,TD}, U, ϕ_eo::WilsonEOPreSpinorfield
) where {TD<:WilsonEOPreDiracOperator}
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    X_eo, Y_eo, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    bc = D.boundary_condition

    clear!(X_eo)
    iters, res = solve_dirac!(X_eo, DdagD, ϕ_eo, Y_eo, temp1, temp2, cg_tol, cg_maxiters) # Y is used here merely as a temp LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!

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

    LinearAlgebra.mul!(Y_eo, D, X_eo)
    mul_oe!(X_eo, U, X_eo, bc, true, Val(1))
    mul_oe!(Y_eo, U, Y_eo, bc, true, Val(-1))
    mul_oo_inv!(X_eo, D.D_oo_inv)
    mul_oo_inv!(Y_eo, D.D_oo_inv)
    add_wilson_eo_derivative!(dU, U, X_eo, Y_eo, bc)

    if has_clover_term(D)
        Xμν = fermion_action.Xμν
        D_oo_inv = D.D_oo_inv
        calc_Xμν_eo_eachsite!(Xμν, X_eo, Y_eo)
        add_clover_derivative!(dU, U, Xμν, -D.csw)
        calc_small_Xμν_eachsite!(Xμν, D_oo_inv)
        add_clover_derivative!(dU, U, Xμν, -2D.csw)
    end

    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::FermionAction{true,1,TD}, U, ϕ_eo::WilsonEOPreSpinorfield
) where {TD<:WilsonEOPreDiracOperator}
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    rhmc = fermion_action.rhmc_info_md
    n = get_n_inverse(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    D_oo_inv = D.D_oo_inv
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
        Xs, shifts, DdagD, ϕ_eo, temp1, temp2, Ys, cg_tol, cg_maxiters
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
        mul_oe!(Xs[i+1], U, Xs[i+1], bc, true, Val(1))
        mul_oe!(Ys[i+1], U, Ys[i+1], bc, true, Val(-1))
        mul_oo_inv!(Xs[i+1], D_oo_inv)
        mul_oo_inv!(Ys[i+1], D_oo_inv)
        add_wilson_derivative!(dU, U, Xs[i+1], Ys[i+1], bc; coeff=coeffs[i])

        if has_clover_term(D)
            Xμν = D.Xμν
            calc_Xμν_eo_eachsite!(Xμν, Xs[i+1], Ys[i+1])
            add_clover_derivative!(dU, U, Xμν, -D.csw; coeff=coeffs[i])
            calc_small_Xμν_eachsite!(Xμν, D_oo_inv)
            add_clover_derivative!(dU, U, Xμν, -2D.csw; coeff=coeffs[i])
        end
    end

    return nothing
end

function add_wilson_eo_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T,M}, X_eo::TF, Y_eo::TF, bc; coeff=1
) where {B,T,M,TF<:WilsonEOPreSpinorfield{B,T,M}}
    fac = T(0.5coeff)
    X = X_eo.parent
    Y = Y_eo.parent
    bulk = eachindex(U)
    itr = eachindex(dU, U, X, Y)

    parallelfor(itr, B, Val(M), (X_eo, Y_eo), (dU,), (dU, U, X, Y)) do site, dU, U, X, Y
        add_wilson_eo_derivative_kernel!(dU, U, X, Y, site, bc, fac, bulk)
    end

    return nothing
end

function add_wilson_eo_derivative_kernel!(dU, U, X_eo, Y_eo, site, bc, fac, bulk)
    # sites that begin with a "_" are meant for indexing into the even-odd preconn'ed
    # fermion field
    NT = size(U, 4)
    _site = map_to_half(site, bulk)
    @nexprs 4 μ -> (
        Nμ = axes(U, μ);
        _siteμ⁺ = map_to_half(move(site, μ, 1, Nμ), bulk);
        X⁺ = apply_bc(X_eo[_siteμ⁺], bc, site, Val(1), NT, Val(μ));
        Y⁺ = apply_bc(Y_eo[_siteμ⁺], bc, site, Val(1), NT, Val(μ));
        B = spintrace(spin_proj(X⁺, Val(-μ)), Y_eo[_site]);
        C = spintrace(spin_proj(Y⁺, Val(μ)), X_eo[_site]);
        dU[μ, site] += fac * traceless_antihermitian(cmatmul_oo(U[μ, site], B + C))
    )
    return nothing
end

function calc_Xμν_eo_eachsite!(
    Xμν::Tensorfield{B,T}, X_eo::TF, Y_eo::TF
) where {B,T,M,TF<:WilsonEOPreSpinorfield{B,T,M}}
    X = X_eo.parent
    Y = Y_eo.parent
    bulk = eachindex(X)

    parallelfor(eachindex(Xμν), B, Val(M), () , (Xμν,), (Xμν, X, Y)) do site, Xμν, X, Y
        calc_Xμν_eo_kernel!(Xμν, X, Y, site, bulk)
    end

    return nothing
end

function calc_Xμν_eo_kernel!(Xμν, X, Y, site, bulk)
    _site = map_to_half(site, bulk)

    X₁₂ =
        spintrace(σμν_spin_mul(X[_site], Val(1), Val(2)), Y[_site]) +
        spintrace(σμν_spin_mul(Y[_site], Val(1), Val(2)), X[_site])
    Xμν[1i32, 2i32, site] = X₁₂
    Xμν[2i32, 1i32, site] = -X₁₂

    X₁₃ =
        spintrace(σμν_spin_mul(X[_site], Val(1), Val(3)), Y[_site]) +
        spintrace(σμν_spin_mul(Y[_site], Val(1), Val(3)), X[_site])
    Xμν[1i32, 3i32, site] = X₁₃
    Xμν[3i32, 1i32, site] = -X₁₃

    X₁₄ =
        spintrace(σμν_spin_mul(X[_site], Val(1), Val(4)), Y[_site]) +
        spintrace(σμν_spin_mul(Y[_site], Val(1), Val(4)), X[_site])
    Xμν[1i32, 4i32, site] = X₁₄
    Xμν[4i32, 1i32, site] = -X₁₄

    X₂₃ =
        spintrace(σμν_spin_mul(X[_site], Val(2), Val(3)), Y[_site]) +
        spintrace(σμν_spin_mul(Y[_site], Val(2), Val(3)), X[_site])
    Xμν[2i32, 3i32, site] = X₂₃
    Xμν[3i32, 2i32, site] = -X₂₃

    X₂₄ =
        spintrace(σμν_spin_mul(X[_site], Val(2), Val(4)), Y[_site]) +
        spintrace(σμν_spin_mul(Y[_site], Val(2), Val(4)), X[_site])
    Xμν[2i32, 4i32, site] = X₂₄
    Xμν[4i32, 2i32, site] = -X₂₄

    X₃₄ =
        spintrace(σμν_spin_mul(X[_site], Val(3), Val(4)), Y[_site]) +
        spintrace(σμν_spin_mul(Y[_site], Val(3), Val(4)), X[_site])
    Xμν[3i32, 4i32, site] = X₃₄
    Xμν[4i32, 3i32, site] = -X₃₄
    return nothing
end

function calc_small_Xμν_eachsite!(
    Xμν::Tensorfield{B,T}, D_oo_inv::Paulifield{B,T,M,true}
) where {B,T,M}
    bulk = eachindex(Xμν)
    itr = eachindex(Xμν, D_oo_inv)

    parallelfor(itr, B, Val(M), (), (Xμν,), (Xμν, D_oo_inv)) do site, Xμν, D_oo_inv
        calc_small_Xμν_kernel!(Xμν, D_oo_inv, site, T, bulk)
    end

    return nothing
end

function calc_small_Xμν_kernel!(Xμν, D_oo_inv, site, ::Type{T}, bulk) where {T}
    if isodd(site)
        _site = map_to_half(site, bulk)
        Minv = D_oo_inv[_site]

        X₁₂ = spintrace_pauli(Minv, Val(1), Val(2))
        Xμν[1i32, 2i32, site] = X₁₂
        Xμν[2i32, 1i32, site] = -X₁₂

        X₁₃ = spintrace_pauli(Minv, Val(1), Val(3))
        Xμν[1i32, 3i32, site] = X₁₃
        Xμν[3i32, 1i32, site] = -X₁₃

        X₁₄ = spintrace_pauli(Minv, Val(1), Val(4))
        Xμν[1i32, 4i32, site] = X₁₄
        Xμν[4i32, 1i32, site] = -X₁₄

        X₂₃ = spintrace_pauli(Minv, Val(2), Val(3))
        Xμν[2i32, 3i32, site] = X₂₃
        Xμν[3i32, 2i32, site] = -X₂₃

        X₂₄ = spintrace_pauli(Minv, Val(2), Val(4))
        Xμν[2i32, 4i32, site] = X₂₄
        Xμν[4i32, 2i32, site] = -X₂₄

        X₃₄ = spintrace_pauli(Minv, Val(3), Val(4))
        Xμν[3i32, 4i32, site] = X₃₄
        Xμν[4i32, 3i32, site] = -X₃₄
    else
        X = zero3(T)
        Xμν[1i32, 2i32, site] = X
        Xμν[2i32, 1i32, site] = X
        Xμν[1i32, 3i32, site] = X
        Xμν[3i32, 1i32, site] = X
        Xμν[1i32, 4i32, site] = X
        Xμν[4i32, 1i32, site] = X
        Xμν[2i32, 3i32, site] = X
        Xμν[3i32, 2i32, site] = X
        Xμν[2i32, 4i32, site] = X
        Xμν[4i32, 2i32, site] = X
        Xμν[3i32, 4i32, site] = X
        Xμν[4i32, 3i32, site] = X
    end
end

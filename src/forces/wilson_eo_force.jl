const WilsonEOPreSpinorfield{B,T,M,A} = SpinorfieldEO{B,T,M,A,4}

function calc_dSfdU!(
    dU, fermion_action::WilsonEOPreFermionAction{2,C}, U, ϕ_eo::WilsonEOPreSpinorfield
) where {C}
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    X_eo, Y_eo, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    bc = D.boundary_condition

    # clear!(X_eo)
    # solve_dirac!(X_eo, DdagD, ϕ_eo, Y_eo, temp1, temp2, cg_tol, cg_maxiters) # Y is used here merely as a temp LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    #
    # LinearAlgebra.mul!(Y_eo, D, X_eo)
    # mul_oe!(X_eo, U, X_eo, bc, true, Val(1)) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    # mul_oe!(Y_eo, U, Y_eo, bc, true, Val(-1)) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    # mul_oo_inv!(X_eo, D.D_oo_inv)
    # mul_oo_inv!(Y_eo, D.D_oo_inv)
    # add_wilson_eo_derivative!(dU, U, X_eo, Y_eo, bc)

    if has_clover_term(D)
        Xμν = fermion_action.Xμν
        D_oo_inv = D.D_oo_inv
        # calc_Xμν_eo_eachsite!(Xμν, X_eo, Y_eo)
        # add_clover_derivative!(dU, U, Xμν, D.csw)
        calc_Xμν_small_eachsite!(Xμν, D_oo_inv)
        add_clover_derivative!(dU, U, Xμν, 2D.csw) 
    end

    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::WilsonEOPreFermionAction{Nf}, U, ϕ::WilsonEOPreSpinorfield
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
        mul_oe!(Xs[i+1], U, Xs[i+1], bc, true, Val(1)) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
        mul_oe!(Ys[i+1], U, Ys[i+1], bc, true, Val(-1)) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
        mul_oo_inv!(Xs[i+1], D.D_oo_inv)
        mul_oo_inv!(Ys[i+1], D.D_oo_inv)
        add_wilson_derivative!(dU, U, Xs[i+1], Ys[i+1], bc; coeff=coeffs[i])

        if has_clover_term(D)
            Xμν = D.Xμν
            calc_Xμν_eachsite!(Xμν, Xs[i+1], Ys[i+1])
            add_clover_derivative!(dU, U, Xμν, D.csw; coeff=coeffs[i])
        end
    end

    # TODO:
    if has_clover_term(D)
        Xμν = D.Xμν
        calc_Xμν_small_eachsite!(Xμν, D_oo_inv)
        add_clover_derivative!(dU, U, Xμν, 2D.csw) 
    end

    return nothing
end

function add_wilson_eo_derivative!(
    dU::Colorfield{CPU,T}, U::Gaugefield{CPU,T}, X_eo::TF, Y_eo::TF, bc; coeff=1
) where {T,TF<:WilsonEOPreSpinorfield{CPU,T}}
    check_dims(dU, U, X_eo, Y_eo)
    fac = T(0.5coeff)

    # INFO: If we write out the kernel and use @batch, the program crashes for some reason
    # Stems from "pload" from StrideArraysCore.jl but ONLY if we write it out AND overload
    # "object_and_preserve" (cant reproduce in MWE yet)
    # is fine, because writing it like this makes the GPU port easier

    #= @batch  =#for site in eachindex(dU)
        add_wilson_eo_derivative_kernel!(dU, U, X_eo, Y_eo, site, bc, fac)
    end

    return nothing
end

function add_wilson_eo_derivative_kernel!(dU, U, X_eo, Y_eo, site, bc⁺, fac)
    # sites that begin with a "_" are meant for indexing into the even-odd preconn'ed
    # fermion field 
    NX, NY, NZ, NT = dims(U)
    NV = NX * NY * NZ * NT
    _site = eo_site(site, NX, NY, NZ, NT, NV)

    _siteμ⁺ = eo_site(move(site, 1, 1, NX), NX, NY, NZ, NT, NV)
    B = spintrace(spin_proj(X_eo[_siteμ⁺], Val(-1)), Y_eo[_site])
    C = spintrace(spin_proj(Y_eo[_siteμ⁺], Val(1)), X_eo[_site])
    dU[1i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[1, site], B + C))

    _siteμ⁺ = eo_site(move(site, 2, 1, NY), NX, NY, NZ, NT, NV)
    B = spintrace(spin_proj(X_eo[_siteμ⁺], Val(-2)), Y_eo[_site])
    C = spintrace(spin_proj(Y_eo[_siteμ⁺], Val(2)), X_eo[_site])
    dU[2i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[2, site], B + C))

    _siteμ⁺ = eo_site(move(site, 3, 1, NZ), NX, NY, NZ, NT, NV)
    B = spintrace(spin_proj(X_eo[_siteμ⁺], Val(-3)), Y_eo[_site])
    C = spintrace(spin_proj(Y_eo[_siteμ⁺], Val(3)), X_eo[_site])
    dU[3i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[3, site], B + C))

    _siteμ⁺ = eo_site(move(site, 4, 1, NT), NX, NY, NZ, NT, NV)
    B = spintrace(spin_proj(X_eo[_siteμ⁺], Val(-4)), Y_eo[_site])
    C = spintrace(spin_proj(Y_eo[_siteμ⁺], Val(4)), X_eo[_site])
    dU[4i32, site] += bc⁺ * fac * traceless_antihermitian(cmatmul_oo(U[4, site], B + C))
    return nothing
end

function calc_Xμν_eo_eachsite!(
    Xμν::Tensorfield{CPU,T}, X_eo::TF, Y_eo::TF
) where {T,TF<:WilsonEOPreSpinorfield}
    X = X_eo.parent
    Y = Y_eo.parent

    @batch for site in eachindex(Xμν)
        calc_Xμν_eo_kernel!(Xμν, X, Y, site)
    end

    return nothing
end

function calc_Xμν_eo_kernel!(Xμν, X, Y, site)
    NX, NY, NZ, NT = dims(Xμν)
    NV = NX * NY * NZ * NT
    _site = eo_site(site, NX, NY, NZ, NT, NV)

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

function calc_Xμν_small_eachsite!(
    Xμν::Tensorfield{CPU,T}, D_oo_inv::WilsonEODiagonal{CPU,T,true}
) where {T}
    check_dims(Xμν, D_oo_inv)

    for site in eachindex(Xμν)
        calc_Xμν_small_kernel!(Xμν, D_oo_inv, site, T)
    end

    return nothing
end

function calc_Xμν_small_kernel!(Xμν, D_oo_inv, site, ::Type{T}) where {T}
    if isodd(site)
        NX, NY, NZ, NT = dims(Xμν)
        NV = NX * NY * NZ * NT
        _site = eo_site_switch(site, NX, NY, NZ, NT, NV)
        A₊ = D_oo_inv[1, _site]
        A₋ = D_oo_inv[2, _site]

        X₁₂ = im * spintrace_σμν(A₊, A₋, Val(1), Val(2))
        Xμν[1i32, 2i32, site] = X₁₂
        Xμν[2i32, 1i32, site] = -X₁₂

        X₁₃ = im * spintrace_σμν(A₊, A₋, Val(1), Val(3))
        Xμν[1i32, 3i32, site] = X₁₃
        Xμν[3i32, 1i32, site] = -X₁₃

        X₁₄ = im * spintrace_σμν(A₊, A₋, Val(1), Val(4))
        Xμν[1i32, 4i32, site] = X₁₄
        Xμν[4i32, 1i32, site] = -X₁₄

        X₂₃ = im * spintrace_σμν(A₊, A₋, Val(2), Val(3))
        Xμν[2i32, 3i32, site] = X₂₃
        Xμν[3i32, 2i32, site] = -X₂₃

        X₂₄ = im * spintrace_σμν(A₊, A₋, Val(2), Val(4))
        Xμν[2i32, 4i32, site] = X₂₄
        Xμν[4i32, 2i32, site] = -X₂₄

        X₃₄ = im * spintrace_σμν(A₊, A₋, Val(3), Val(4))
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

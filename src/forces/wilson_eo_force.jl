function calc_dSfdU!(
    dU, fermion_action::FermionAction{false,2,TD}, U, ϕ_eo::WilsonEOPreSpinorfield
) where {TD<:WilsonEOPreDiracOperator}
    clear!(dU)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    bc = D.boundary_condition
    solver_md = fermion_action.solver_md
    tol, maxiters, datafile = get_info(solver_md)

    if isnothing(fermion_action.D_low)
        X, temps... = fermion_action.temps[1:4]
        clear!(X)
        solve_dirac!(X, DdagD, ϕ_eo, temps; tol, maxiters, datafile)
    else
        # TODO: delta = solver_action.delta
        X, temps... = fermion_action.temps[1:3]
        clear!(X)
        U_low, temps_low... = fermion_action.temps_low[1:6]
        copy!(U_low, U)
        D_low = fermion_action.D_low(U_low)
        DdagD_low = DdaggerD(D_low)
        solve_dirac_mixed!(
            X, DdagD, DdagD_low, ϕ_eo, temps, temps_low; tol, maxiters, datafile
        )
    end

    LinearAlgebra.mul!(temps[1], D, X)
    mul_oe!(X, U, X, bc, true, Val(1))
    mul_oe!(temps[1], U, temps[1], bc, true, Val(-1))
    mul_oo_inv!(X, D.D_oo_inv)
    mul_oo_inv!(temps[1], D.D_oo_inv)
    add_wilson_eo_derivative!(dU, U, X, temps[1], bc)

    if has_clover_term(D)
        Xμν = fermion_action.Xμν
        D_oo_inv = D.D_oo_inv
        calc_Xμν_eo_eachsite!(Xμν, X, temps[1])
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
    rhmc = fermion_action.rhmc_info_md
    n = get_n_inverse(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    D_oo_inv = D.D_oo_inv
    bc = D.boundary_condition
    solver_md = fermion_action.solver_md
    tol, maxiters, datafile = get_info(solver_md)

    shifts = get_β_inverse(rhmc)
    coeffs = get_α_inverse(rhmc)

    if isnothing(fermion_action.D_low)
        Xs = fermion_action.temps[1:n+1]
        Ys = fermion_action.temps[n+2:2n+2]
        temps = fermion_action.temps[2n+3:2n+4]

        for X in Xs
            clear!(X)
        end

        solve_dirac_multishift!(
            Xs, shifts, DdagD, ϕ_eo, temps, Ys; tol, maxiters, datafile
        )
    else
        # TODO: delta = solver_action.delta
        Xs = fermion_action.temps[1:n+1]
        temps = fermion_action.temps[n+2:n+3]
        U_low = fermion_action.temps_low[1]
        Xs_low = fermion_action.temps_low[2:n+2]
        ps_low = fermion_action.temps_low[n+3:2n+3]
        temps_low = fermion_action.temps_low[2n+4:2n+6]
        copy!(U_low, U)
        D_low = fermion_action.D_low(U_low)
        DdagD_low = DdaggerD(D_low)

        for X in Xs
            clear!(X)
        end

        solve_dirac_multishift_mixed!(
            Xs, shifts, DdagD, DdagD_low, ϕ_eo, temps,
            Xs_low, ps_low, temps_low;
            tol, maxiters, datafile
        )
    end

    for i in 1:n
        LinearAlgebra.mul!(temps[1], D, Xs[i+1]) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
        mul_oe!(Xs[i+1], U, Xs[i+1], bc, true, Val(1))
        mul_oe!(temps[1], U, temps[1], bc, true, Val(-1))
        mul_oo_inv!(Xs[i+1], D_oo_inv)
        mul_oo_inv!(temps[1], D_oo_inv)
        add_wilson_derivative!(dU, U, Xs[i+1], temps[1], bc; coeff=coeffs[i])

        if has_clover_term(D)
            Xμν = D.Xμν
            calc_Xμν_eo_eachsite!(Xμν, Xs[i+1], temps[1])
            add_clover_derivative!(dU, U, Xμν, -D.csw; coeff=coeffs[i])
            calc_small_Xμν_eachsite!(Xμν, D_oo_inv)
            add_clover_derivative!(dU, U, Xμν, -2D.csw; coeff=coeffs[i])
        end
    end

    return nothing
end

function add_wilson_eo_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,TU,M}, X_eo::TF, Y_eo::TF, bc; coeff=1
) where {B,T,M,TU,TF<:WilsonEOPreSpinorfield{B,TU,M}}
    fac = T(0.5coeff)
    X = X_eo.parent
    Y = Y_eo.parent
    itr = eachindex(dU, U, X, Y)
    padded_bulk = dU.topology.bulk_sites_padded

    parallelfor(itr, B, Val(M), (X_eo, Y_eo), (dU,), (dU, U, X, Y)) do site, (dU, U, X, Y)
        add_wilson_eo_derivative_kernel!(dU, U, X, Y, site, bc, fac, padded_bulk)
    end

    return nothing
end

function add_wilson_eo_derivative_kernel!(dU, U, X_eo, Y_eo, site, bc, fac, padded_bulk)
    # sites that begin with a "_" are meant for indexing into the even-odd preconn'ed
    # fermion field
    NT = size(U, 4)
    _site = map_to_half(site, padded_bulk)
    @inbounds begin
        @nexprs 4 μ -> (
            Nμ = axes(U, μ);
            _siteμ⁺ = map_to_half(move(site, μ, 1, Nμ), padded_bulk);
            X⁺ = apply_bc(X_eo[_siteμ⁺], bc, site, Val(1), NT, Val(μ));
            Y⁺ = apply_bc(Y_eo[_siteμ⁺], bc, site, Val(1), NT, Val(μ));
            B = spintrace(spin_proj(X⁺, Val(-μ)), Y_eo[_site]);
            C = spintrace(spin_proj(Y⁺, Val(μ)), X_eo[_site]);
            dU[μ, site] += fac * traceless_antihermitian(cmatmul_oo(U[μ, site], B + C))
        )
    end

    return nothing
end

function calc_Xμν_eo_eachsite!(
    Xμν::Tensorfield{B,T}, X_eo::TF, Y_eo::TF
) where {B,T,M,TU,TF<:WilsonEOPreSpinorfield{B,TU,M}}
    X = X_eo.parent
    Y = Y_eo.parent
    padded_bulk = X.topology.bulk_sites_padded

    parallelfor(eachindex(Xμν), B, Val(M), () , (Xμν,), (Xμν, X, Y)) do site, (Xμν, X, Y)
        calc_Xμν_eo_kernel!(Xμν, X, Y, site, padded_bulk)
    end

    return nothing
end

function calc_Xμν_eo_kernel!(Xμν, X, Y, site, padded_bulk)
    _site = map_to_half(site, padded_bulk)
    @inbounds begin
        Xn = X[_site]
        Yn = Y[_site]
        @nexprs 6 i -> (
            Xμν[i, site] = spintrace(σμν_spin_mul(Xn, Val(i)), Yn) +
                spintrace(σμν_spin_mul(Yn, Val(i)), Xn);
        )
    end

    return nothing
end

function calc_small_Xμν_eachsite!(
    Xμν::Tensorfield{B,T}, D_oo_inv::Paulifield{B,TU,M,true}
) where {B,T,M,TU}
    itr = eachindex(Xμν, D_oo_inv)
    padded_bulk = Xμν.topology.bulk_sites_padded

    parallelfor(itr, B, Val(M), (), (Xμν,), (Xμν, D_oo_inv)) do site, (Xμν, D_oo_inv)
        calc_small_Xμν_kernel!(Xμν, D_oo_inv, site, T, padded_bulk)
    end

    return nothing
end

#= @inline  =#function calc_small_Xμν_kernel!(Xμν, D_oo_inv, site, ::Type{T}, padded_bulk) where {T}
    @inbounds begin
        if isodd(site)
            _site = map_to_half(site, padded_bulk)
            Minv = D_oo_inv[_site]
            @nexprs 6 i -> (
                Xᵢ = spintrace_pauli(Minv, Val(i));
                Xμν[i, site] = Xᵢ
            )
        else
            X = zero3(T)
            @nexprs 6 i -> (
                Xμν[i, site] = X
            )
        end
    end

    return nothing
end

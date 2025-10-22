function calc_dSfdU!(
    dU, fermion_action::FermionAction{false,2,TD}, U, ϕ::WilsonSpinorfield
) where {TD<:WilsonDiracOperator}
    clear!(dU)
    X, Y, temp1, temp2 = fermion_action.temps[1:4]
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    bc = D.boundary_condition
    solver_md = fermion_action.solver_md
    tol, maxiters, datafile = get_info(solver_md)

    clear!(X)
    solve_dirac!(X, DdagD, ϕ, Y, temp1, temp2; tol, maxiters, datafile)

    LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    add_wilson_derivative!(dU, U, X, Y, bc)

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
    rhmc = fermion_action.rhmc_info_md
    n = get_n_inverse(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    bc = D.boundary_condition
    temp1, temp2 = fermion_action.temps[1:2]
    Xs = fermion_action.temps[3:n+3]
    Ys = fermion_action.temps[n+4:2n+4]
    solver_md = fermion_action.solver_md
    tol, maxiters, datafile = get_info(solver_md)

    for X in Xs
        clear!(X)
    end

    shifts = get_β_inverse(rhmc)
    coeffs = get_α_inverse(rhmc)
    solve_dirac_multishift!(
        Xs, shifts, DdagD, ϕ, temp1, temp2, Ys; tol, maxiters, datafile
    )

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
    dU::Colorfield{B,T}, U::Gaugefield{B,TU,M}, X::TF, Y::TF, bc; coeff=1
) where {B,T,M,TU,TF<:WilsonSpinorfield{B,TU,M}}
    fac = T(0.5coeff)
    itr = eachindex(dU, U, X, Y)

    parallelfor(itr, B, Val(M), (X, Y), (dU,), (dU, U, X, Y)) do site, (dU, U, X, Y)
        add_wilson_derivative_kernel!(dU, U, X, Y, site, bc, fac)
    end

    return nothing
end

function add_wilson_derivative_kernel!(dU, U, X, Y, site, bc, fac)
    NT = size(dU, 4)

    @inbounds begin
        Xn = X[site]
        Yn = Y[site]
        @nexprs 4 μ -> (
            Nμ = axes(U, μ);
            siteμ⁺ = move(site, μ, 1, Nμ);
            X⁺ = apply_bc(X[siteμ⁺], bc, site, Val(1), NT, Val(μ));
            Y⁺ = apply_bc(Y[siteμ⁺], bc, site, Val(1), NT, Val(μ));
            B = spintrace(spin_proj(X⁺, Val(-μ)), Yn);
            C = spintrace(spin_proj(Y⁺, Val(μ)), Xn);
            dU[μ, site] += fac * traceless_antihermitian(cmatmul_oo(U[μ, site], B + C))
        )
    end

    return nothing
end

function add_clover_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T,M}, Xμν::Tensorfield{B,T,M}, csw; coeff=1
) where {B,T,M}
    fac = T(csw * coeff / 2)
    itr = eachindex(dU, U, Xμν)

    parallelfor(itr, B, Val(M), (U, Xμν), (dU,), (dU, U, Xμν); do_edges=Val(true)) do site, (dU, U, Xμν)
        add_clover_derivative_kernel!(dU, U, Xμν, site, fac, T)
    end

    return nothing
end

function add_clover_derivative_kernel!(dU, U, Xμν, site, fac, ::Type{T}) where {T}
    tmp =
        Xμν∇Fμν(Xμν, U, 1, 2, site, T) +
        Xμν∇Fμν(Xμν, U, 1, 3, site, T) +
        Xμν∇Fμν(Xμν, U, 1, 4, site, T)
    @inbounds dU[1, site] += fac * traceless_antihermitian(cmatmul_oo(U[1, site], tmp))

    tmp =
        Xμν∇Fμν(Xμν, U, 2, 1, site, T) +
        Xμν∇Fμν(Xμν, U, 2, 3, site, T) +
        Xμν∇Fμν(Xμν, U, 2, 4, site, T)
    @inbounds dU[2, site] += fac * traceless_antihermitian(cmatmul_oo(U[2, site], tmp))

    tmp =
        Xμν∇Fμν(Xμν, U, 3, 1, site, T) +
        Xμν∇Fμν(Xμν, U, 3, 2, site, T) +
        Xμν∇Fμν(Xμν, U, 3, 4, site, T)
    @inbounds dU[3, site] += fac * traceless_antihermitian(cmatmul_oo(U[3, site], tmp))

    tmp =
        Xμν∇Fμν(Xμν, U, 4, 1, site, T) +
        Xμν∇Fμν(Xμν, U, 4, 2, site, T) +
        Xμν∇Fμν(Xμν, U, 4, 3, site, T)
    @inbounds dU[4, site] += fac * traceless_antihermitian(cmatmul_oo(U[4, site], tmp))
    return nothing
end

function calc_Xμν_wilson_eachsite!(
    Xμν::Tensorfield{B,T}, X::TF, Y::TF
) where {B,T,M,TF<:WilsonSpinorfield{B,T,M}}
    parallelfor(eachindex(Xμν, X, Y), B, Val(M), (), (Xμν,), (Xμν, X, Y)) do site, (Xμν, X, Y)
        calc_Xμν_wilson_kernel!(Xμν, X, Y, site)
    end

    return nothing
end

function calc_Xμν_wilson_kernel!(Xμν, X, Y, site)
    @inbounds begin
        Xn = X[site]
        Yn = Y[site]
        @nexprs 6 i -> (
            Xμν[i, site] = spintrace(σμν_spin_mul(Xn, Val(i)), Yn) +
                spintrace(σμν_spin_mul(Yn, Val(i)), Xn)
        )
    end

    return nothing
end

function Xμν∇Fμν(Xμν, U, μ, ν, site, ::Type{T}) where {T}
    Nμ = axes(U, μ)
    Nν = axes(U, ν)
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteν⁺ = move(site, ν, 1, Nν)
    siteν⁻ = move(site, ν, -1, Nν)
    siteμ⁺ν⁺ = move(siteμ⁺, ν, 1, Nν)
    siteμ⁺ν⁻ = move(siteμ⁺, ν, -1, Nν)
    i = get_tensor_index(μ, ν)
    sgn = μ > ν ? -1 : 1

    # get reused matrices up to cache (can precalculate some products too)
    # Uνsiteμ⁺ = U[ν,siteμ⁺]
    # Uμsiteν⁺ = U[μ,siteν⁺]
    # Uνsite = U[ν,site]
    # Uνsiteμ⁺ν⁻ = U[ν,siteμ⁺ν⁻]
    # Uμsiteν⁻ = U[μ,siteν⁻]
    # Uνsiteν⁻ = U[ν,siteν⁻]

    @inbounds begin
        component =
            cmatmul_oddo(U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site], Xμν[i, site]) +
            cmatmul_odod(U[ν, siteμ⁺], U[μ, siteν⁺], Xμν[i, siteν⁺], U[ν, site]) +
            cmatmul_oodd(U[ν, siteμ⁺], Xμν[i, siteμ⁺ν⁺], U[μ, siteν⁺], U[ν, site]) +
            cmatmul_oodd(Xμν[i, siteμ⁺], U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site]) -
            cmatmul_ddoo(U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻], Xμν[i, site]) -
            cmatmul_ddoo(U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], Xμν[i, siteν⁻], U[ν, siteν⁻]) -
            cmatmul_dodo(U[ν, siteμ⁺ν⁻], Xμν[i, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻]) -
            cmatmul_oddo(Xμν[i, siteμ⁺], U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻])
    end

    return im * T(sgn / 8) * component
end

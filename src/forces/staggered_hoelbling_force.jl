function calc_dSfdU!( # Force for unrooted Staggered-Hoelbling Action (Nf=2)
    dU,
    fermion_action::FermionAction{false,2,TD},
    U,
    ϕ::StaggeredSpinorfield,
) where {TD<:StaggeredHoelblingDiracOperator}
    clear!(dU)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    term = get_mass_term(D)
    bc = D.boundary_condition
    solver_md = fermion_action.solver_md
    tol, maxiters, datafile = get_info(solver_md)

    if isnothing(fermion_action.D_low)
        X, temps... = fermion_action.temps[1:4]
        clear!(X)
        solve_dirac!(X, DdagD, ϕ, temps; tol, maxiters, datafile)
    else
        # TODO: delta = solver_action.delta
        X, temps... = fermion_action.temps[1:3]
        clear!(X)
        U_low, temps_low... = fermion_action.temps_low[1:6]
        copy!(U_low, U)
        D_low = fermion_action.D_low(U_low)
        DdagD_low = DdaggerD(D_low)
        solve_dirac_mixed!(
            X, DdagD, DdagD_low, ϕ, temps, temps_low; tol, maxiters, datafile
        )
    end

    LinearAlgebra.mul!(temps[1], D, X)
    add_staggered_hoelbling_derivative!(dU, U, X, temps[1], bc, term)
    return nothing
end

function calc_dSfdU!( # Force for single flavor Staggered-Hoelbling Action (Nf=2)
    dU,
    fermion_action::FermionAction{true,1,TD},
    U,
    ϕ::StaggeredSpinorfield,
) where {TD<:StaggeredHoelblingDiracOperator}
    clear!(dU)
    rhmc = fermion_action.rhmc_info_md
    n = get_n_inverse(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    term = get_mass_term(D)
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
            Xs, shifts, DdagD, ϕ, temps, Ys; tol, maxiters, datafile
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
            Xs, shifts, DdagD, DdagD_low, ϕ, temps,
            Xs_low, ps_low, temps_low;
            tol, maxiters, datafile
        )
    end

    for i in 1:n
        LinearAlgebra.mul!(temps[1], D, Xs[i+1])
        add_staggered_hoelbling_derivative!(dU, U, Xs[i+1], temps[1], bc, term; coeff=coeffs[i])
    end

    return nothing
end

function add_staggered_hoelbling_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,TU,M}, X::TF, Y::TF, bc, term; coeff=1
) where {B,T,M,TU,TF<:StaggeredSpinorfield{B,TU,M}}
    fac1 = T(-0.5coeff)
    fac2 = T(coeff)
    _μ, _ν, _ρ, _σ = term
    itr = eachindex(dU, U, X, Y)

    parallelfor(itr, B, Val(M), (U, X, Y), (dU,), (dU, U, X, Y); do_edges=Val(true)) do site, (dU, U, X, Y)
        add_staggered_derivative_kernel!(dU, U, X, Y, site, bc, fac1, T)
        add_hoelbling_derivative_kernel!(dU, _μ, _ν, U, X, Y, site, bc, fac2)
        add_hoelbling_derivative_kernel!(dU, _ρ, _σ, U, X, Y, site, bc, fac2)
    end

    return nothing
end

function add_hoelbling_derivative_kernel!(
    dU, ::Val{μ}, ::Val{ν}, U, X, Y, site, bc, fac::T
) where {μ,ν,T}
    B = Y∇MμνX(X, Y, U, Val(μ), Val(ν), site, bc, T)
    C = Y∇MμνX(Y, X, U, Val(μ), Val(ν), site, bc, T)
    @inbounds dU[μ, site] += fac * traceless_antihermitian(cmatmul_oo(U[μ, site], B + C))

    B = Y∇MμνX(X, Y, U, Val(ν), Val(μ), site, bc, T)
    C = Y∇MμνX(Y, X, U, Val(ν), Val(μ), site, bc, T)
    @inbounds dU[ν, site] += fac * traceless_antihermitian(cmatmul_oo(U[ν, site], B + C))
    return nothing
end

function Y∇MμνX(X, Y, U, ::Val{μ}, ::Val{ν}, site, bc, ::Type{T}) where {μ,ν,T}
    NT = size(U, 4)
    Nμ = axes(U, μ)
    Nν = axes(U, ν)
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteν⁻ = move(site, ν, -1, Nν)
    siteν⁺ = move(site, ν, 1, Nν)
    siteμ⁺ν⁺ = move(siteμ⁺, ν, 1, Nν)
    siteμ⁺ν⁻ = move(siteμ⁺, ν, -1, Nν)

    @inbounds begin
        # Start
        η1 = im * T(1 / 8 * staggered_ημν(Val(μ), Val(ν), site, T, Val(true)))
        η2 = im * T(1 / 8 * staggered_ημν(Val(μ), Val(ν), siteν⁺, T, Val(true)))
        η3 = im * T(1 / 8 * staggered_ημν(Val(μ), Val(ν), siteν⁻, T, Val(true)))
        Y1 = η1 * Y[site]
        Y2 = η2 * apply_bc(Y[siteν⁺], bc, site, Val(1), NT, Val(ν))
        Y3 = η3 * apply_bc(Y[siteν⁻], bc, site, Val(-1), NT, Val(ν))

        # Stop
        X1 = apply_bc(
            apply_bc(X[siteμ⁺ν⁺], bc, site, Val(1), NT, Val(μ)), bc, site, Val(1), NT, Val(ν)
        )
        X2 = apply_bc(
            apply_bc(X[siteμ⁺ν⁻], bc, site, Val(1), NT, Val(μ)), bc, site, Val(-1), NT, Val(ν)
        )
        X3 = apply_bc(X[siteμ⁺], bc, site, Val(1), NT, Val(μ))

        out =
            cmatmul_oo(U[ν, siteμ⁺], ckron(X1, Y1)) + # site -> siteμ+ -> siteμ+ν+
            cmatmul_do(U[ν, siteμ⁺ν⁻], ckron(X2, Y1)) + # site -> siteμ+ -> siteμ+ν-
            cmatmul_od(ckron(X3, Y2), U[ν, site]) + # siteν+ -> site -> siteμ+
            cmatmul_oo(ckron(X3, Y3), U[ν, siteν⁻]) # siteν- -> site -> siteμ+
    end

    return out
end

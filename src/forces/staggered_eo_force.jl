function calc_dSfdU!(
    dU, fermion_action::FermionAction{false,4,TD}, U, ϕ_eo::StaggeredEOPreSpinorfield
) where {TD<:StaggeredEOPreDiracOperator}
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

    clear!(temps[1])
    mul_oe!(temps[1], U, X, bc, true, false)
    add_staggered_eo_derivative!(dU, U, X, temps[1], bc)
    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::FermionAction{true,Nf,TD}, U, ϕ_eo::StaggeredEOPreSpinorfield
) where {Nf,TD<:StaggeredEOPreDiracOperator}
    clear!(dU)
    rhmc = fermion_action.rhmc_info_md
    n = get_n_inverse(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
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
        mul_oe!(temps[1], U, Xs[i+1], bc, true, false)
        add_staggered_eo_derivative!(dU, U, Xs[i+1], temps[1], bc; coeff=coeffs[i])
    end

    return nothing
end

function add_staggered_eo_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,TU,M}, X_eo::TF, Y_eo::TF, bc; coeff=1
) where {B,T,M,TU,TF<:StaggeredEOPreSpinorfield{B,TU,M}}
    X = X_eo.parent
    Y = Y_eo.parent
    fac = T(-0.5coeff)
    itr = eachindex(dU, U, X, Y)
    padded_bulk = dU.topology.bulk_sites_padded

    parallelfor(itr, B, Val(M), (X_eo, Y_eo), (dU,), (dU, U, X, Y)) do site, (dU, U, X, Y)
        add_staggered_eo_derivative_kernel!(dU, U, X, Y, site, bc, fac, padded_bulk, T)
    end

    return nothing
end

@inline function add_staggered_eo_derivative_kernel!(
    dU, U, X, Y, site, bc, fac, padded_bulk, ::Type{T}
) where {T}
    # sites that begin with a "_" are meant for indexing into the even-odd preconn'ed
    # fermion field
    NT = size(U, 4)
    _site = map_to_half(site, padded_bulk)

    # use @nexprs here to statically generate the loop
    # this makes it so Val(μ) is well defined at each iteration and no type-instabilities arise
    @inbounds begin
        @nexprs 4 μ -> (
            Nμ = axes(U, μ);
            _siteμ⁺ = map_to_half(move(site, μ, 1, Nμ), padded_bulk);
            η = staggered_η(Val(μ), site, T);
            B = ckron(apply_bc(X[_siteμ⁺], bc, site, Val(1), NT, Val(μ)), Y[_site]);
            C = ckron(apply_bc(Y[_siteμ⁺], bc, site, Val(1), NT, Val(μ)), X[_site]);
            dU[μ, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[μ, site], B - C))
        )
    end

    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::FermionAction{false,8,TD}, U, ϕ::StaggeredSpinorfield
) where {TD<:StaggeredDiracOperator}
    clear!(dU)
    X, Y, temp1, temp2 = fermion_action.temps[1:4]
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    bc = D.boundary_condition
    solver_action = fermion_action.solver_action
    tol, maxiters, datafile = get_info(solver_action)

    clear!(X) # initial guess is zero
    solve_dirac!(X, DdagD, ϕ, Y, temp1, temp2; tol, maxiters, datafile)

    LinearAlgebra.mul!(Y, D, X)
    add_staggered_derivative!(dU, U, X, Y, bc)
    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::FermionAction{true,Nf,TD}, U, ϕ::StaggeredSpinorfield
) where {Nf,TD<:StaggeredDiracOperator}
    clear!(dU)
    rhmc = fermion_action.rhmc_info_md
    n = get_n_inverse(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    bc = D.boundary_condition
    temp1, temp2 = fermion_action.temps[1:2]
    Xs = fermion_action.temps[3:n+3]
    Ys = fermion_action.temps[n+4:2n+4]
    solver_action = fermion_action.solver_action
    tol, maxiters, datafile = get_info(solver_action)

    for X in Xs
        clear!(X)
    end

    shifts = get_β_inverse(rhmc)
    coeffs = get_α_inverse(rhmc)
    solve_dirac_multishift!(
        Xs, shifts, DdagD, ϕ, temp1, temp2, Ys; tol, maxiters, datafile
    )

    for i in 1:n
        LinearAlgebra.mul!(Ys[i+1], D, Xs[i+1])
        add_staggered_derivative!(dU, U, Xs[i+1], Ys[i+1], bc; coeff=coeffs[i])
    end

    return nothing
end

function add_staggered_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T,M}, X::TF, Y::TF, bc; coeff=1
) where {B,T,M,TF<:StaggeredSpinorfield{B,T,M}}
    fac = T(-0.5coeff)
    itr = eachindex(dU, U, X, Y)

    parallelfor(itr, B, Val(M), (X, Y), (dU,), (dU, U, X, Y)) do site, (dU, U, X, Y)
        add_staggered_derivative_kernel!(dU, U, X, Y, site, bc, fac, T)
    end

    return nothing
end

@inline function add_staggered_derivative_kernel!(dU, U, X, Y, site, bc, fac, ::Type{T}) where {T}
    NT = size(U, 4)

    # use @nexprs here to statically generate the loop
    # this makes it so Val(μ) is well defined at each iteration and no type-instabilities arise
    @inbounds begin
        @nexprs 4 μ -> (
            Nμ = axes(U, μ);
            siteμ⁺ = move(site, μ, 1, Nμ);
            η = staggered_η(Val(μ), site, T);
            B = ckron(apply_bc(X[siteμ⁺], bc, site, Val(1), NT, Val(μ)), Y[site]);
            C = ckron(apply_bc(Y[siteμ⁺], bc, site, Val(1), NT, Val(μ)), X[site]);
            dU[μ, site] += (fac * η) * traceless_antihermitian(cmatmul_oo(U[μ, site], B - C))
        )
    end

    return nothing
end

const WilsonFermionfield{B,T,A} = Fermionfield{B,T,A,4}

function calc_dSfdU!(dU, fermion_action::WilsonFermionAction{2}, U, ϕ::WilsonFermionfield)
    @assert dims(dU) == dims(U) == dims(ϕ)
    clear!(dU)
    cg_tol = fermion_action.cg_tol
    cg_maxiters = fermion_action.cg_maxiters
    X, Y, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)

    clear!(X)
    solve_D⁻¹x!(X, DdagD, ϕ, Y, temp1, temp2, cg_tol, cg_maxiters) # Y is used here merely as a temp LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    add_wilson_derivative!(dU, U, X, Y, D.anti_periodic)
    D.csw != 0 && add_clover_derivative!(dU, U, X, Y, D.csw)
    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::WilsonFermionAction{Nf}, U, ϕ::WilsonFermionfield
) where {Nf}
    @assert dims(dU) == dims(U) == dims(ϕ)
    clear!(dU)
    cg_tol = fermion_action.cg_tol
    cg_maxiters = fermion_action.cg_maxiters
    rhmc = fermion_action.rhmc_info_md
    n = rhmc.coeffs.n
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    Xs = fermion_action.rhmc_temps1
    Ys = fermion_action.rhmc_temps2
    temp1, temp2 = fermion_action.cg_temps

    for v in Xs
        clear!(v)
    end
    shifts = rhmc.coeffs_inverse.β
    coeffs = rhmc.coeffs_inverse.α
    solve_D⁻¹x_multishift!(Xs, shifts, DdagD, ϕ, temp1, temp2, Ys, cg_tol, cg_maxiters)

    for i in 1:n
        LinearAlgebra.mul!(Ys[i+1], D, Xs[i+1]) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
        add_wilson_derivative!(dU, U, Xs[i+1], Ys[i+1], D.anti_periodic; coeff=coeffs[i])
        if D.csw != 0
            add_clover_derivative!(dU, U, Xs[i+1], Ys[i+1], D.csw; coeff=coeffs[i])
        end
    end
    return nothing
end

function add_wilson_derivative!(
    dU, U, X::T, Y::T, anti; coeff=1
) where {T<:WilsonFermionfield}
    @assert dims(dU) == dims(U) == dims(X) == dims(Y)
    NT = dims(U)[4]
    fac = float_type(U)(0.5coeff)

    # If we write out the kernel and use @batch, the program crashes...
    # Happens in "pload" from StrideArraysCore.jl but ONLY if we write it out AND overload
    # "object_and_preserve" (I will not give up on Polyester though)
    @batch for site in eachindex(dU)
        bc⁺ = boundary_factor(anti, site[4], 1, NT)
        wilson_derivative_kernel!(dU, U, X, Y, site, bc⁺, fac)
    end
end

function add_clover_derivative!(
    dU, U, X::T, Y::T, csw; coeff=1
) where {T<:WilsonFermionfield}
    @assert dims(dU) == dims(U) == dims(X) == dims(Y)
    Tf = float_type(U)
    fac = Tf(-csw * coeff / 2)

    @batch for site in eachindex(dU)
        clover_derivative_kernel!(dU, U, X, Y, site, fac, Tf)
    end
end

function wilson_derivative_kernel!(dU, U, X, Y, site, bc⁺, fac)
    NX, NY, NZ, NT = dims(U)
    siteμ⁺ = move(site, 1, 1, NX)
    B = spintrace(spin_proj(X[siteμ⁺], Val(-1)), Y[site])
    C = spintrace(spin_proj(Y[siteμ⁺], Val(1)), X[site])
    dU[1, site] = fac * traceless_antihermitian(cmatmul_oo(U[1, site], B + C))

    siteμ⁺ = move(site, 2, 1, NY)
    B = spintrace(spin_proj(X[siteμ⁺], Val(-2)), Y[site])
    C = spintrace(spin_proj(Y[siteμ⁺], Val(2)), X[site])
    dU[2, site] = fac * traceless_antihermitian(cmatmul_oo(U[2, site], B + C))

    siteμ⁺ = move(site, 3, 1, NZ)
    B = spintrace(spin_proj(X[siteμ⁺], Val(-3)), Y[site])
    C = spintrace(spin_proj(Y[siteμ⁺], Val(3)), X[site])
    dU[3, site] = fac * traceless_antihermitian(cmatmul_oo(U[3, site], B + C))

    siteμ⁺ = move(site, 4, 1, NT)
    B = spintrace(spin_proj(X[siteμ⁺], Val(-4)), Y[site])
    C = spintrace(spin_proj(Y[siteμ⁺], Val(4)), X[site])
    dU[4, site] = bc⁺ * fac * traceless_antihermitian(cmatmul_oo(U[4, site], B + C))
    return nothing
end

function clover_derivative_kernel!(dU, U, X, Y, site, fac, ::Type{T}) where {T}
    tmp =
        Xμν∇Fμν(X, Y, U, Val(1), Val(2), site) +
        Xμν∇Fμν(X, Y, U, Val(1), Val(3), site) +
        Xμν∇Fμν(X, Y, U, Val(1), Val(4), site)
    dU[1, site] += fac * traceless_antihermitian(cmatmul_oo(U[1, site], tmp))

    tmp =
        Xμν∇Fμν(X, Y, U, Val(1), Val(2), site) +
        Xμν∇Fμν(X, Y, U, Val(2), Val(3), site) +
        Xμν∇Fμν(X, Y, U, Val(2), Val(4), site)
    dU[2, site] += fac * traceless_antihermitian(cmatmul_oo(U[2, site], tmp))

    tmp =
        Xμν∇Fμν(X, Y, U, Val(1), Val(3), site) +
        Xμν∇Fμν(X, Y, U, Val(2), Val(3), site) +
        Xμν∇Fμν(X, Y, U, Val(3), Val(4), site)
    dU[3, site] += fac * traceless_antihermitian(cmatmul_oo(U[3, site], tmp))

    tmp =
        Xμν∇Fμν(X, Y, U, Val(1), Val(4), site) +
        Xμν∇Fμν(X, Y, U, Val(2), Val(4), site) +
        Xμν∇Fμν(X, Y, U, Val(3), Val(4), site)
    dU[4, site] += fac * traceless_antihermitian(cmatmul_oo(U[4, site], tmp))
    return nothing
end

function Xμν∇Fμν(X, Y, U, ::Val{μ}, ::Val{ν}, site) where {μ,ν}
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
    XY_site = spintrace(σμν_spin_mul(X[site], Val(μ), Val(ν)), Y[site])
    XY_siteν⁺ = spintrace(σμν_spin_mul(X[siteν⁺], Val(μ), Val(ν)), Y[siteν⁺])
    XY_siteμ⁺ν⁺ = spintrace(σμν_spin_mul(X[siteμ⁺ν⁺], Val(μ), Val(ν)), Y[siteμ⁺ν⁺])
    XY_siteμ⁺ = spintrace(σμν_spin_mul(X[siteμ⁺], Val(μ), Val(ν)), Y[siteμ⁺])
    XY_siteν⁻ = spintrace(σμν_spin_mul(X[siteν⁻], Val(μ), Val(ν)), Y[siteν⁻])
    XY_siteμ⁺ν⁻ = spintrace(σμν_spin_mul(X[siteμ⁺ν⁻], Val(μ), Val(ν)), Y[siteμ⁺ν⁻])

    component =
        cmatmul_oddo(U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site], XY_site) +
        cmatmul_odod(U[ν, siteμ⁺], U[μ, siteν⁺], XY_siteν⁺, U[ν, site]) +
        cmatmul_oodd(U[ν, siteμ⁺], XY_siteμ⁺ν⁺, U[μ, siteν⁺], U[ν, site]) +
        cmatmul_oodd(XY_siteμ⁺, U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site]) -
        cmatmul_ddoo(U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻], XY_site) -
        cmatmul_ddoo(U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], XY_siteν⁻, U[ν, siteν⁻]) -
        cmatmul_dodo(U[ν, siteμ⁺ν⁻], XY_siteμ⁺ν⁻, U[μ, siteν⁻], U[ν, siteν⁻]) -
        cmatmul_oddo(XY_siteμ⁺, U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻])

    return (im//4) * component
end

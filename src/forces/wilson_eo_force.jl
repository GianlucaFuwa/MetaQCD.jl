const WilsonEOPreFermionfield{B,T,A} = EvenOdd{B,T,A,4}

function calc_dSfdU!(
    dU, fermion_action::WilsonEOPreFermionAction{2,C}, U, ϕ_eo::WilsonEOPreFermionfield
) where {C}
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    X_eo, Y_eo, temp1, temp2 = fermion_action.cg_temps
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    anti = D.anti_periodic

    clear!(X_eo)
    solve_dirac!(X_eo, DdagD, ϕ_eo, Y_eo, temp1, temp2, cg_tol, cg_maxiters) # Y is used here merely as a temp LinearAlgebra.mul!(Y, D, X) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!

    # set odd components of X and Y
    LinearAlgebra.mul!(Y_eo, D, X_eo)
    mul_oe!(X_eo, U, X_eo, anti, true, Val(1)) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    mul_oe!(Y_eo, U, Y_eo, anti, true, Val(1)) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
    mul_oo_inv!(X_eo, D.D_oo_inv)
    mul_oo_inv!(Y_eo, D.D_oo_inv)
    add_wilson_eo_derivative!(dU, U, X_eo, Y_eo, anti)

    if C
        Xμν = fermion_action.Xμν
        calc_Xμν_eachsite!(Xμν, X_eo, Y_eo)
        add_clover_eo_derivative!(dU, U, Xμν, D.csw)
    end
    return nothing
end

function calc_dSfdU!(
    dU, fermion_action::WilsonEOPreFermionAction{Nf,C}, U, ϕ::WilsonEOPreFermionfield
) where {Nf,C}
    clear!(dU)
    cg_tol = fermion_action.cg_tol_md
    cg_maxiters = fermion_action.cg_maxiters_md
    rhmc = fermion_action.rhmc_info_md
    n = get_n(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    anti = D.anti_periodic
    Xs = fermion_action.rhmc_temps1[1:n+1]
    Ys = fermion_action.rhmc_temps2[1:n+1]
    temp1, temp2 = fermion_action.cg_temps

    for X in Xs
        clear!(X)
    end

    shifts = get_β_inverse(rhmc)
    coeffs = get_α_inverse(rhmc)
    solve_dirac_multishift!(Xs, shifts, DdagD, ϕ, temp1, temp2, Ys, cg_tol, cg_maxiters)

    for i in 1:n
        LinearAlgebra.mul!(Ys[i+1], D, Xs[i+1]) # Need to prefix with LinearAlgebra to avoid ambiguity with Gaugefields.mul!
        add_wilson_derivative!(dU, U, Xs[i+1], Ys[i+1], anti; coeff=coeffs[i])
        if C
            Xμν = fermion_action.Xμν
            calc_Xμν_eachsite!(Xμν, Xs[i+1], Ys[i+1])
            add_clover_derivative!(dU, U, Xμν, D.csw; coeff=coeffs[i])
        end
    end
    return nothing
end

function add_wilson_eo_derivative!(
    dU::Colorfield{CPU,T}, U::Gaugefield{CPU,T}, X_eo::TF, Y_eo::TF, anti; coeff=1
) where {T,TF<:WilsonEOPreFermionfield{CPU,T}}
    check_dims(dU, U, X_eo, Y_eo)
    NT = dims(U)[4]
    fac = T(0.5coeff)

    # If we write out the kernel and use @batch, the program crashes for some reason
    # Stems from "pload" from StrideArraysCore.jl but ONLY if we write it out AND overload
    # "object_and_preserve" (cant reproduce in MWE yet)
    # is fine, because writing it like this makes the GPU port easier
    #= @batch  =#for site in eachindex(dU)
        bc⁺ = boundary_factor(anti, site[4], 1, NT)
        if isodd(site)
            add_wilson_eo_derivative_kernel!(dU, U, X_eo, Y_eo, site, bc⁺, fac, Val(-1))
        else
            add_wilson_eo_derivative_kernel!(dU, U, Y_eo, X_eo, site, bc⁺, fac, Val(1))
        end
    end
end

function add_wilson_eo_derivative_kernel!(
    dU, U, X_eo, Y_eo, site, bc⁺, fac, ::Val{DIR}
) where {DIR}
    NX, NY, NZ, NT = dims(U)
    NV = NX * NY * NZ * NT
    _site = eo_site(site, NX, NY, NZ, NT, NV)

    _siteμ⁺ = eo_site(move(site, 1, 1, NX), NX, NY, NZ, NT, NV)
    # B = spintrace(spin_proj(X_eo[_siteμ⁺], Val(-1)), Y_eo[_site])
    C = spintrace(spin_proj(Y_eo[_siteμ⁺], Val(1DIR)), X_eo[_site])
    dU[1i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[1, site], C))

    _siteμ⁺ = eo_site(move(site, 2, 1, NY), NX, NY, NZ, NT, NV)
    # B = spintrace(spin_proj(X_eo[_siteμ⁺], Val(-2)), Y_eo[site])
    C = spintrace(spin_proj(Y_eo[_siteμ⁺], Val(2DIR)), X_eo[site])
    dU[2i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[2, site], C))

    _siteμ⁺ = eo_site(move(site, 3, 1, NZ), NX, NY, NZ, NT, NV)
    # B = spintrace(spin_proj(X_eo[_siteμ⁺], Val(-3)), Y_eo[site])
    C = spintrace(spin_proj(Y_eo[_siteμ⁺], Val(3DIR)), X_eo[site])
    dU[3i32, site] += fac * traceless_antihermitian(cmatmul_oo(U[3, site], C))

    _siteμ⁺ = eo_site(move(site, 4, 1, NT), NX, NY, NZ, NT, NV)
    # B = spintrace(spin_proj(X_eo[_siteμ⁺], Val(-4)), Y_eo[site])
    C = spintrace(spin_proj(Y_eo[_siteμ⁺], Val(4DIR)), X_eo[site])
    dU[4i32, site] += bc⁺ * fac * traceless_antihermitian(cmatmul_oo(U[4, site], C))
    return nothing
end

function add_clover_eo_derivative!(
    dU::Colorfield{CPU,T}, U::Gaugefield{CPU,T}, Xμν::Tensorfield{CPU,T}, csw; coeff=1
) where {T}
    check_dims(dU, U, Xμν)
    fac = T(csw * coeff / 2)

    #= @batch  =#for site in eachindex(dU)
        add_clover_eo_derivative_kernel!(dU, U, Xμν, site, fac, T)
    end

    return nothing
end

function add_clover_eo_derivative_kernel!(dU, U, Xμν, site, fac, ::Type{T}) where {T}
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
    Xμν::Tensorfield{CPU,T}, X_eo::TF, Y_eo::TF
) where {T,TF<:WilsonEOPreFermionfield}
    check_dims(Xμν, X_eo, Y_eo)
    even = true

    #= @batch  =#for site in eachindex(even, Xμν)
        calc_Xμν_kernel!(Xμν, X_eo, Y_eo, site)
    end

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


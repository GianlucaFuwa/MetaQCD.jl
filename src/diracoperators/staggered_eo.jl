"""
    StaggeredEOPreDiracOperator(f::AbstractField, mass; bc_str="antiperiodic")
    StaggeredEOPreDiracOperator(
        D::Union{StaggeredDiracOperator,StaggeredEOPreDiracOperator},
        U::Gaugefield
    )

Create a free even-odd preconditioned Staggered Dirac Operator with mass `mass`.

`bc_str` can either be `"periodic"` or `"antiperiodic"` and specifies the boundary
condition in the time direction.

If `csw ≠ 0`, a clover term is included. 

This object cannot be directly applied to a fermion vector, since it lacks a gauge
background.
A Wilson Dirac operator with gauge background is created by applying it to a `Gaugefield`
`U` like `D_gauge = D(U)`

# Type Parameters:
- `B`: Backend (CPU / CUDA / ROCm)
- `T`: Floating point precision
- `TF`: Type of the `Spinorfield` used to store intermediate results when using the 
        Hermitian version of the operator
- `TG`: Type of the underlying `Gaugefield`
- `BC`: Boundary Condition in time direction
"""
struct StaggeredEOPreDiracOperator{B,T,TF,TG,BC} <: AbstractDiracOperator
    U::TG
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    mass::Float64
    boundary_condition::BC # Only in time direction
    function StaggeredEOPreDiracOperator(
        f::AbstractField{B,T}, mass; bc_str="antiperiodic", kwargs...
    ) where {B,T}
        U = nothing
        temp = even_odd(Spinorfield(f; staggered=true))
        TG = Nothing
        TF = typeof(temp)
        boundary_condition = create_bc(bc_str, f.topology)
        BC = typeof(boundary_condition)
        return new{B,T,TF,TG,BC}(U, temp, mass, boundary_condition)
    end

    function StaggeredEOPreDiracOperator(
        D::StaggeredEOPreDiracOperator{B,T,TF}, U::Gaugefield{B,T}
    ) where {B,T,TF}
        check_dims(U, D.temp.parent)
        TG = typeof(U)
        BC = typeof(D.boundary_condition)
        return new{B,T,TF,TG,BC}(U, D.temp, D.mass, D.boundary_condition)
    end
end

function (D::StaggeredEOPreDiracOperator{B,T})(U::Gaugefield{B,T}) where {B,T}
    return StaggeredEOPreDiracOperator(D, U)
end

struct StaggeredEOPreFermionAction{Nf,TD,CT,RI1,RI2,RT} <: AbstractFermionAction{Nf}
    D::TD
    cg_temps::CT
    rhmc_info_action::RI1
    rhmc_info_md::RI2
    rhmc_temps1::RT # this holds the results of multishift cg
    rhmc_temps2::RT # this holds the basis vectors in multishift cg
    cg_tol_action::Float64
    cg_tol_md::Float64
    cg_maxiters_action::Int64
    cg_maxiters_md::Int64
    function StaggeredEOPreFermionAction(
        f::AbstractField,
        mass;
        bc_str="antiperiodic",
        Nf=4,
        rhmc_spectral_bound=(mass^2, 6.0),
        rhmc_order_action=15,
        rhmc_prec_action=42,
        rhmc_order_md=10,
        rhmc_prec_md=42,
        cg_tol_action=1e-14,
        cg_tol_md=1e-12,
        cg_maxiters_action=1000,
        cg_maxiters_md=1000,
        kwargs...,
    )
        D = StaggeredEOPreDiracOperator(f, mass; bc_str=bc_str)
        TD = typeof(D)
        rhmc_lambda_low = rhmc_spectral_bound[1]
        rhmc_lambda_high = rhmc_spectral_bound[2]

        if Nf == 4
            cg_temps = ntuple(_ -> even_odd(Spinorfield(f; staggered=true)), 4)
            power = Nf//8
            rhmc_info_action = RHMCParams(
                power;
                n=rhmc_order_action,
                precision=rhmc_prec_action,
                lambda_low=rhmc_lambda_low,
                lambda_high=rhmc_lambda_high,
            )
            n_temps = rhmc_order_action
            rhmc_temps1 = ntuple(
                _ -> even_odd(Spinorfield(f; staggered=true)), n_temps + 1
            )
            rhmc_temps2 = ntuple(
                _ -> even_odd(Spinorfield(f; staggered=true)), n_temps + 1
            )
            rhmc_info_md = nothing
        else
            @assert 4 > Nf > 0 "Nf should be between 1 and 4 (was $Nf)"
            cg_temps = ntuple(_ -> even_odd(Spinorfield(f; staggered=true)), 2)
            power = Nf//8
            rhmc_info_action = RHMCParams(
                power;
                n=rhmc_order_action,
                precision=rhmc_prec_action,
                lambda_low=rhmc_lambda_low,
                lambda_high=rhmc_lambda_high,
            )
            power = Nf//4
            rhmc_info_md = RHMCParams(
                power;
                n=rhmc_order_md,
                precision=rhmc_prec_md,
                lambda_low=rhmc_lambda_low,
                lambda_high=rhmc_lambda_high,
            )
            n_temps = max(rhmc_order_md, rhmc_order_action)
            rhmc_temps1 = ntuple(
                _ -> even_odd(Spinorfield(f; staggered=true)), n_temps + 1
            )
            rhmc_temps2 = ntuple(
                _ -> even_odd(Spinorfield(f; staggered=true)), n_temps + 1
            )
        end

        CT = typeof(cg_temps)
        RI1 = typeof(rhmc_info_action)
        RI2 = typeof(rhmc_info_md)
        RT = typeof(rhmc_temps1)
        return new{Nf,TD,CT,RI1,RI2,RT}(
            D,
            cg_temps,
            rhmc_info_action,
            rhmc_info_md,
            rhmc_temps1,
            rhmc_temps2,
            cg_tol_action,
            cg_tol_md,
            cg_maxiters_action,
            cg_maxiters_md,
        )
    end
end

function calc_fermion_action(
    fermion_action::StaggeredEOPreFermionAction{4},
    U::Gaugefield,
    ϕ_eo::StaggeredEOPreSpinorfield,
)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψ_eo, temp1, temp2, temp3 = fermion_action.cg_temps
    cg_tol = fermion_action.cg_tol_action
    cg_maxiters = fermion_action.cg_maxiters_action

    clear!(ψ_eo) # initial guess is zero
    solve_dirac!(ψ_eo, DdagD, ϕ_eo, temp1, temp2, temp3, cg_tol, cg_maxiters) # ψ = (D†D)⁻¹ϕ
    Sf = dot(ϕ_eo, ψ_eo)
    return distributed_reduce(real(Sf), +, U)
end

function calc_fermion_action(
    fermion_action::StaggeredEOPreFermionAction{Nf},
    U::Gaugefield,
    ϕ_eo::StaggeredEOPreSpinorfield,
) where {Nf}
    cg_tol = fermion_action.cg_tol_action
    cg_maxiters = fermion_action.cg_maxiters_action
    rhmc = fermion_action.rhmc_info_action
    n = get_n(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψs = fermion_action.rhmc_temps1[1:n+1]
    ps = fermion_action.rhmc_temps2[1:n+1]
    temp1, temp2 = fermion_action.cg_temps

    for v_eo in ψs
        clear!(v_eo)
    end

    shifts = get_β_inverse(rhmc)
    coeffs = get_α_inverse(rhmc)
    α₀ = get_α0_inverse(rhmc)
    solve_dirac_multishift!(ψs, shifts, DdagD, ϕ_eo, temp1, temp2, ps, cg_tol, cg_maxiters)
    ψ_eo = ψs[1]
    clear!(ψ_eo) # D⁻¹ϕ doesn't appear in the partial fraction decomp so we can use it to sum

    axpy!(α₀, ϕ_eo, ψ_eo)

    for i in 1:n
        axpy!(coeffs[i], ψs[i+1], ψ_eo)
    end

    Sf = dot(ψ_eo, ψ_eo)
    return distributed_reduce(real(Sf), +, U)
end

function sample_pseudofermions!(
    ϕ_eo, fermion_action::StaggeredEOPreFermionAction{Nf}, U
) where {Nf}
    cg_tol = fermion_action.cg_tol_action
    cg_maxiters = fermion_action.cg_maxiters_action
    rhmc = fermion_action.rhmc_info_action
    n = get_n(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψs = fermion_action.rhmc_temps1[1:n+1]
    ps = fermion_action.rhmc_temps2[1:n+1]
    temp1, temp2 = fermion_action.cg_temps

    for v in ψs
        clear!(v)
    end

    shifts = get_β(rhmc)
    coeffs = get_α(rhmc)
    α₀ = get_α0(rhmc)
    gaussian_pseudofermions!(ϕ_eo) # D⁻¹ϕ doesn't appear in the partial fraction decomp so we can use it to sum
    solve_dirac_multishift!(ψs, shifts, DdagD, ϕ_eo, temp1, temp2, ps, cg_tol, cg_maxiters)

    mul!(ϕ_eo, α₀)

    for i in 1:n
        axpy!(coeffs[i], ψs[i+1], ϕ_eo)
    end

    return nothing
end

function solve_dirac!(
    ψ_eo, D::T, ϕ_eo, temp1, temp2, temp3, temp4, temp5; tol=1e-14, maxiters=1000
) where {T<:StaggeredEOPreDiracOperator}
    error("Not implemented yet")
    # TODO: CGNE
    return nothing
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ψ_eo::TF, D::DdaggerD{StaggeredEOPreDiracOperator{B,T,TF,TG,BC}}, ϕ_eo::TF
) where {B,T,TF,TG,BC}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass = T(D.parent.mass)
    bc = D.parent.boundary_condition

    # ψₒ = Dₒₑϕₑ
    mul_oe!(ψ_eo, U, ϕ_eo, bc, true, false)
    # ψₑ = DₑₒDₒₑϕₑ
    mul_eo!(ψ_eo, U, ψ_eo, bc, false, false)
    axpby!(mass^2, ϕ_eo, -1, ψ_eo) # ψₑ = m²ϕₑ - DₑₒDₒₑϕₑ
    return nothing
end

function mul_oe!(
    ψ_eo::TF, U::Gaugefield{CPU,T}, ϕ_eo::TF, bc, into_odd, dagg::Bool; fac=1
) where {T,TF<:SpinorfieldEO{CPU,T}}
    check_dims(ψ_eo, ϕ_eo, U)
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    fdims = dims(ψ)
    NV = ψ.NV

    @batch for site in eachindex(ψ)
        isodd(site) || continue
        _site = if into_odd
            eo_site(site, fdims..., NV)
        else
            eo_site_switch(site, fdims..., NV)
        end
        ψ[_site] = fac * staggered_eo_kernel(U, ϕ, site, bc, T, dagg)
    end

    update_halo!(ψ) # TODO: Even-odd halo exchange
    return nothing
end

function mul_eo!(
    ψ_eo::TF, U::Gaugefield{CPU,T}, ϕ_eo::TF, bc, into_odd, dagg::Bool; fac=1
) where {T,TF<:SpinorfieldEO{CPU,T}}
    check_dims(ψ_eo, ϕ_eo, U)
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    fdims = dims(ψ)
    NV = ψ.NV

    @batch for site in eachindex(ψ)
        iseven(site) || continue
        _site = if into_odd
            eo_site_switch(site, fdims..., NV)
        else
            eo_site(site, fdims..., NV)
        end
        ψ[_site] = fac * staggered_eo_kernel(U, ϕ, site, bc, T, dagg)
    end

    update_halo!(ψ) # TODO: Even-odd halo exchange
    return nothing
end

function staggered_eo_kernel(U, ϕ, site, bc, ::Type{T}, dagg::Bool) where {T}
    sgn = dagg ? -1 : 1
    # sites that begin with a "_" are meant for indexing into the even-odd preconn'ed
    # fermion field 
    NX, NY, NZ, NT = dims(U)
    NV = NX * NY * NZ * NT
    ψₙ = zero(ϕ[site])
    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    _siteμ⁺ = eo_site(move(site, 1, 1, NX), NX, NY, NZ, NT, NV)
    siteμ⁻ = move(site, 1, -1, NX)
    _siteμ⁻ = eo_site(siteμ⁻, NX, NY, NZ, NT, NV)
    η = sgn * staggered_η(Val(1), site)
    ψₙ += η * cmvmul(U[1, site], ϕ[_siteμ⁺])
    ψₙ -= η * cmvmul_d(U[1, siteμ⁻], ϕ[_siteμ⁻])

    _siteμ⁺ = eo_site(move(site, 2, 1, NY), NX, NY, NZ, NT, NV)
    siteμ⁻ = move(site, 2, -1, NY)
    _siteμ⁻ = eo_site(siteμ⁻, NX, NY, NZ, NT, NV)
    η = sgn * staggered_η(Val(2), site)
    ψₙ += η * cmvmul(U[2, site], ϕ[_siteμ⁺])
    ψₙ -= η * cmvmul_d(U[2, siteμ⁻], ϕ[_siteμ⁻])

    _siteμ⁺ = eo_site(move(site, 3, 1, NZ), NX, NY, NZ, NT, NV)
    siteμ⁻ = move(site, 3, -1, NZ)
    _siteμ⁻ = eo_site(siteμ⁻, NX, NY, NZ, NT, NV)
    η = sgn * staggered_η(Val(3), site)
    ψₙ += η * cmvmul(U[3, site], ϕ[_siteμ⁺])
    ψₙ -= η * cmvmul_d(U[3, siteμ⁻], ϕ[_siteμ⁻])

    _siteμ⁺ = eo_site(move(site, 4, 1, NT), NX, NY, NZ, NT, NV)
    siteμ⁻ = move(site, 4, -1, NT)
    _siteμ⁻ = eo_site(siteμ⁻, NX, NY, NZ, NT, NV)
    η = sgn * staggered_η(Val(4), site)
    ψₙ += η * cmvmul(U[4, site], apply_bc(ϕ[_siteμ⁺], bc, site, Val(1), NT))
    ψₙ -= η * cmvmul_d(U[4, siteμ⁻], apply_bc(ϕ[_siteμ⁻], bc, site, Val(-1), NT))
    return T(0.5) * ψₙ
end

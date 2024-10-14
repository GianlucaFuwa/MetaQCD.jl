"""
    StaggeredDiracOperator(f::AbstractField, mass; bc_str="antiperiodic")
    StaggeredDiracOperator(D::StaggeredDiracOperator, U::Gaugefield)

Create a free Staggered Dirac Operator with mass `mass`.

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
struct StaggeredDiracOperator{B,T,TF,TG,BC} <: AbstractDiracOperator
    U::TG
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    mass::Float64
    boundary_condition::BC # Only in time direction
    function StaggeredDiracOperator(
        f::AbstractField{B,T}, mass; bc_str="antiperiodic", kwargs...
    ) where {B,T}
        U = nothing
        temp = Spinorfield(f; staggered=true)
        TG = Nothing
        TF = typeof(temp)
        boundary_condition = create_bc(bc_str, f.topology)
        BC = typeof(boundary_condition)
        return new{B,T,TF,TG,BC}(U, temp, mass, boundary_condition)
    end

    function StaggeredDiracOperator(
        D::StaggeredDiracOperator{B,T,TF}, U::Gaugefield{B,T}
    ) where {B,T,TF}
        TG = typeof(U)
        BC = typeof(D.boundary_condition)
        return new{B,T,TF,TG,BC}(U, D.temp, D.mass, D.boundary_condition)
    end
end

function (D::StaggeredDiracOperator{B,T})(U::Gaugefield{B,T}) where {B,T}
    return StaggeredDiracOperator(D, U)
end

struct StaggeredFermionAction{R,Nf,TD,CT,RI1,RI2,RT} <: AbstractFermionAction{R,Nf}
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
    function StaggeredFermionAction(
        f::AbstractField,
        mass;
        bc_str="antiperiodic",
        Nf=8,
        rhmc_spectral_bound=(mass^2, 6.0),
        rhmc_order_md=10,
        rhmc_prec_md=42,
        rhmc_order_action=15,
        rhmc_prec_action=42,
        cg_tol_action=1e-14,
        cg_tol_md=1e-12,
        cg_maxiters_action=1000,
        cg_maxiters_md=1000,
        kwargs...,
    )
        D = StaggeredDiracOperator(f, mass; bc_str=bc_str)
        TD = typeof(D)

        if Nf == 8
            R = false
            rhmc_info_action = nothing
            rhmc_info_md = nothing
            rhmc_temps1 = nothing
            rhmc_temps2 = nothing
            cg_temps = ntuple(_ -> Spinorfield(f; staggered=true), 4)
        else
            @assert 8 > Nf > 0 "Nf should be between 1 and 8 (was $Nf)"
            R = true
            rhmc_lambda_low = rhmc_spectral_bound[1]
            rhmc_lambda_high = rhmc_spectral_bound[2]
            cg_temps = ntuple(_ -> Spinorfield(f; staggered=true), 2)
            power = Nf//16
            rhmc_info_action = RHMCParams(
                power;
                n=rhmc_order_action,
                precision=rhmc_prec_action,
                lambda_low=rhmc_lambda_low,
                lambda_high=rhmc_lambda_high,
            )
            power = Nf//8
            rhmc_info_md = RHMCParams(
                power;
                n=rhmc_order_md,
                precision=rhmc_prec_md,
                lambda_low=rhmc_lambda_low,
                lambda_high=rhmc_lambda_high,
            )
            n_temps = max(rhmc_order_md, rhmc_order_action)
            rhmc_temps1 = ntuple(_ -> Spinorfield(f; staggered=true), n_temps + 1)
            rhmc_temps2 = ntuple(_ -> Spinorfield(f; staggered=true), n_temps + 1)
        end

        CT = typeof(cg_temps)
        RI1 = typeof(rhmc_info_action)
        RI2 = typeof(rhmc_info_md)
        RT = typeof(rhmc_temps1)
        return new{R,Nf,TD,CT,RI1,RI2,RT}(
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

function solve_dirac!(
    ψ, D::T, ϕ, temp1, temp2, temp3, temp4, temp5; tol=1e-14, maxiters=1000
) where {T<:StaggeredDiracOperator}
    bicg_stab!(ψ, D, ϕ, temp1, temp2, temp3, temp4, temp5; tol=tol, maxiters=maxiters)
    return nothing
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ψ::TF, D::StaggeredDiracOperator{CPU,T,TF,TG}, ϕ::TF
) where {T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass = T(D.mass)
    bc = D.boundary_condition
    check_dims(ψ, ϕ, U)

    @batch for site in eachindex(ψ)
        ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, T, false)
    end

    update_halo!(ψ)
    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{StaggeredDiracOperator{CPU,T,TF,TG,BC}}, ϕ::TF
) where {T,TF,TG,BC}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass = T(D.parent.mass)
    bc = D.parent.boundary_condition
    check_dims(ψ, ϕ, U)

    @batch for site in eachindex(ψ)
        ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, T, true)
    end

    update_halo!(ψ)
    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::DdaggerD{StaggeredDiracOperator{B,T,TF,TG,BC}}, ϕ::TF
) where {B,T,TF,TG,BC}
    temp = D.parent.temp
    mul!(temp, D.parent, ϕ) # temp = Dϕ
    mul!(ψ, adjoint(D.parent), temp) # ψ = D†Dϕ
    return nothing
end

function staggered_kernel(U, ϕ, site, mass, bc, ::Type{T}, dagg::Bool) where {T}
    sgn = dagg ? -1 : 1
    NX, NY, NZ, NT = dims(U)
    ψₙ = 2mass * ϕ[site]
    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    siteμ⁺ = move(site, 1, 1, NX)
    siteμ⁻ = move(site, 1, -1, NX)
    η = sgn * staggered_η(Val(1), site)
    ψₙ += η * cmvmul(U[1, site], ϕ[siteμ⁺])
    ψₙ -= η * cmvmul_d(U[1, siteμ⁻], ϕ[siteμ⁻])

    siteμ⁺ = move(site, 2, 1, NY)
    siteμ⁻ = move(site, 2, -1, NY)
    η = sgn * staggered_η(Val(2), site)
    ψₙ += η * cmvmul(U[2, site], ϕ[siteμ⁺])
    ψₙ -= η * cmvmul_d(U[2, siteμ⁻], ϕ[siteμ⁻])

    siteμ⁺ = move(site, 3, 1, NZ)
    siteμ⁻ = move(site, 3, -1, NZ)
    η = sgn * staggered_η(Val(3), site)
    ψₙ += η * cmvmul(U[3, site], ϕ[siteμ⁺])
    ψₙ -= η * cmvmul_d(U[3, siteμ⁻], ϕ[siteμ⁻])

    siteμ⁺ = move(site, 4, 1, NT)
    siteμ⁻ = move(site, 4, -1, NT)
    η = sgn * staggered_η(Val(4), site)
    ψₙ += η * cmvmul(U[4, site], apply_bc(ϕ[siteμ⁺], bc, site, Val(1), NT))
    ψₙ -= η * cmvmul_d(U[4, siteμ⁻], apply_bc(ϕ[siteμ⁻], bc, site, Val(-1), NT))
    return T(0.5) * ψₙ
end

# Use Val to reduce the amount of if-statements in the kernel
@inline staggered_η(::Val{1}, site) = 1
@inline staggered_η(::Val{2}, site) = ifelse(iseven(site[1]), 1, -1)
@inline staggered_η(::Val{3}, site) = ifelse(iseven(site[1] + site[2]), 1, -1)
@inline staggered_η(::Val{4}, site) = ifelse(iseven(site[1] + site[2] + site[3]), 1, -1)

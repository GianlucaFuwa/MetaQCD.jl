"""
    StaggeredHoelblingDiracOperator{MT}(f::AbstractField, mass; bc_str="antiperiodic")
    StaggeredHoelblingDiracOperator(D::StaggeredHoelblingDiracOperator, U::Gaugefield)

Create a free Hölbling mass split Staggered Dirac Operator (arXiv:1009.5362) with mass
`mass`.
The type-parameter `MT` determines the kind of operator that is used:
- `MT = 1`: M12 + M34
- `MT = 2`: M13 + M24

`bc_str` can either be `"periodic"` or `"antiperiodic"` and specifies the boundary
condition in the time direction.

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
struct StaggeredHoelblingDiracOperator{MT,B,T,TF,TG,BC} <: AbstractDiracOperator
    U::TG
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    mass::Float64
    c1::Float64
    c2::Float64
    boundary_condition::BC # Only in time direction
    function StaggeredHoelblingDiracOperator{MT}(
        f::AbstractField{B,T}, mass; bc_str="antiperiodic", c1=1.0, c2=1.0, kwargs...
    ) where {MT,B,T}
        @assert MT ∈ (1, 2) "Only 2 Mass Term modes supported so far"
        U = nothing
        temp = Spinorfield(f; staggered=true)
        TG = Nothing
        TF = typeof(temp)
        boundary_condition = create_bc(bc_str, f.topology)
        BC = typeof(boundary_condition)
        return new{MT,B,T,TF,TG,BC}(U, temp, mass, c1, c2, boundary_condition)
    end

    function StaggeredHoelblingDiracOperator(
        D::StaggeredHoelblingDiracOperator{MT,B,T,TF,TG,BC}, U::Gaugefield{B,T}
    ) where {MT,B,T,TF,TG,BC}
        TG_new = typeof(U)
        return new{MT,B,T,TF,TG_new,BC}(U, D.temp, D.mass, D.c1, D.c2, D.boundary_condition)
    end
end

@inline function get_mass_term(::Val{MT}) where {MT}
    return if MT == 1
        (Val(1), Val(2), Val(3), Val(4))
    elseif MT == 2
        (Val(1), Val(3), Val(2), Val(4))
    end
end

@inline function get_mass_term(::StaggeredHoelblingDiracOperator{MT}) where {MT}
    return get_mass_term(Val(MT))
end

function (D::StaggeredHoelblingDiracOperator{MT,B,T})(U::Gaugefield{B,T}) where {MT,B,T}
    return StaggeredHoelblingDiracOperator(D, U)
end

struct StaggeredHoelblingFermionAction{R,Nf,TD,CT,RI1,RI2,RT} <:
    AbstractFermionAction{R,Nf}
    D::TD
    cg_temps::CT
    rhmc_info_action::RI1
    rhmc_info_md::RI2
    rhmc_temps1::RT # this holds the results of multishift cg
    rhmc_temps2::RT # this holds the basis vectors in multishift cg
    # X̃::TX # Used for force calculation
    # Ỹ::TX # Used for force calculation
    cg_tol_action::Float64
    cg_tol_md::Float64
    cg_maxiters_action::Int64
    cg_maxiters_md::Int64
    function StaggeredHoelblingFermionAction{MT}( # INFO: MT: Mass Term
        f::AbstractField,
        mass;
        c1=1.0,
        c2=1.0,
        bc_str="antiperiodic",
        Nf=2,
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
    ) where {MT}
        D = StaggeredHoelblingDiracOperator{MT}(f, mass; c1=c1, c2=c2, bc_str=bc_str)
        TD = typeof(D)

        if Nf == 2
            R = false
            rhmc_info_action = nothing
            rhmc_info_md = nothing
            rhmc_temps1 = nothing
            rhmc_temps2 = nothing
            cg_temps = ntuple(_ -> Spinorfield(f; staggered=true), 4)
        else
            @assert Nf == 1 """
            Nf should be 1 or 2 (was $Nf). If you want Nf > 2, use multiple actions
            """
            R = true
            rhmc_lambda_low = rhmc_spectral_bound[1]
            rhmc_lambda_high = rhmc_spectral_bound[2]
            cg_temps = ntuple(_ -> Spinorfield(f; staggered=true), 2)
            power = Nf//4
            rhmc_info_action = RHMCParams(
                power;
                n=rhmc_order_action,
                precision=rhmc_prec_action,
                lambda_low=rhmc_lambda_low,
                lambda_high=rhmc_lambda_high,
            )
            power = Nf//2
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

        # X̃ = Spinorfield(f; staggered=true)
        # Ỹ = Spinorfield(f; staggered=true)
        CT = typeof(cg_temps)
        RI1 = typeof(rhmc_info_action)
        RI2 = typeof(rhmc_info_md)
        RT = typeof(rhmc_temps1)
        # TX = typeof(X̃)
        return new{R,Nf,TD,CT,RI1,RI2,RT}(
            D,
            cg_temps,
            rhmc_info_action,
            rhmc_info_md,
            rhmc_temps1,
            rhmc_temps2,
            # X̃,
            # Ỹ,
            cg_tol_action,
            cg_tol_md,
            cg_maxiters_action,
            cg_maxiters_md,
        )
    end
end

function solve_dirac!(
    ψ, D::T, ϕ, temp1, temp2, temp3, temp4, temp5; tol=1e-14, maxiters=1000
) where {T<:StaggeredHoelblingDiracOperator}
    bicg_stab!(ψ, D, ϕ, temp1, temp2, temp3, temp4, temp5; tol=tol, maxiters=maxiters)
    return nothing
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ψ::TF, D::StaggeredHoelblingDiracOperator{MT,CPU,T,TF,TG}, ϕ::TF
) where {MT,T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass = T(D.mass)
    term = get_mass_term(D)
    bc = D.boundary_condition
    check_dims(ψ, ϕ, U)

    @batch for site in eachindex(ψ)
        ψ[site] = staggered_hoelbling_kernel(U, ϕ, site, mass, bc, term, T, false)
    end

    update_halo!(ψ)
    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{StaggeredHoelblingDiracOperator{MT,CPU,T,TF,TG,BC}}, ϕ::TF
) where {MT,T,TF,TG,BC}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass = T(D.parent.mass)
    term = get_mass_term(D.parent)
    bc = D.parent.boundary_condition
    check_dims(ψ, ϕ, U)

    @batch for site in eachindex(ψ)
        ψ[site] = staggered_hoelbling_kernel(U, ϕ, site, mass, bc, term, T, true)
    end

    update_halo!(ψ)
    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::DdaggerD{StaggeredHoelblingDiracOperator{MT,B,T,TF,TG,BC}}, ϕ::TF
) where {MT,B,T,TF,TG,BC}
    temp = D.parent.temp
    mul!(temp, D.parent, ϕ) # temp = Dϕ
    mul!(ψ, adjoint(D.parent), temp) # ψ = D†Dϕ
    return nothing
end

function staggered_hoelbling_kernel(U, ϕ, site, mass, bc, term, ::Type{T}, dagg::Bool) where {T}
    sgn = dagg ? -1 : 1
    NX, NY, NZ, NT = dims(U)
    _μ, _ν, _ρ, _σ = term
    ψₙ = (2mass + 4) * ϕ[site] + (
        hoelbling_mass(_μ, _ν, U, ϕ, site, bc, T) +
        hoelbling_mass(_ρ, _σ, U, ϕ, site, bc, T)
    )

    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    siteμ⁺ = move(site, 1, 1, NX)
    siteμ⁻ = move(site, 1, -1, NX)
    η = sgn * staggered_η(Val(1), site)
    ψₙ += η * (cmvmul(U[1, site], ϕ[siteμ⁺]) - cmvmul_d(U[1, siteμ⁻], ϕ[siteμ⁻]))

    siteμ⁺ = move(site, 2, 1, NY)
    siteμ⁻ = move(site, 2, -1, NY)
    η = sgn * staggered_η(Val(2), site)
    ψₙ += η * (cmvmul(U[2, site], ϕ[siteμ⁺]) - cmvmul_d(U[2, siteμ⁻], ϕ[siteμ⁻]))

    siteμ⁺ = move(site, 3, 1, NZ)
    siteμ⁻ = move(site, 3, -1, NZ)
    η = sgn * staggered_η(Val(3), site)
    ψₙ += η * (cmvmul(U[3, site], ϕ[siteμ⁺]) - cmvmul_d(U[3, siteμ⁻], ϕ[siteμ⁻]))

    siteμ⁺ = move(site, 4, 1, NT)
    siteμ⁻ = move(site, 4, -1, NT)
    η = sgn * staggered_η(Val(4), site)
    ϕ⁺ = apply_bc(ϕ[siteμ⁺], bc, site, Val(1), NT)
    ϕ⁻ = apply_bc(ϕ[siteμ⁻], bc, site, Val(-1), NT)
    ψₙ += η * (cmvmul(U[4, site], ϕ⁺) - cmvmul_d(U[4, siteμ⁻], ϕ⁻))
    return T(0.5) * ψₙ
end

function hoelbling_mass(::Val{μ}, ::Val{ν}, U, ϕ, site, bc, ::Type{T}) where {μ,ν,T}
    # XXX:Assuming μ < ν
    Nμ = dims(U)[μ]
    Nν = dims(U)[ν]
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteμ⁻ = move(site, μ, -1, Nμ)
    siteν⁺ = move(site, ν, 1, Nν)
    siteν⁻ = move(site, ν, -1, Nν)
    siteμ⁺ν⁺ = move(siteμ⁺, ν, 1, Nν)
    siteμ⁺ν⁻ = move(siteμ⁺, ν, -1, Nν)
    siteμ⁻ν⁺ = move(siteμ⁻, ν, 1, Nν)
    siteμ⁻ν⁻ = move(siteμ⁻, ν, -1, Nν)

    tmpϕ = apply_bc(
        apply_bc(ϕ[siteμ⁺ν⁺], bc, site, Val(1), Nμ, Val(μ)),
        bc, site, Val(1), Nν, Val(ν)
    )
    tmp = cmatmul_oo(U[μ, site], U[ν, siteμ⁺]) + cmatmul_oo(U[ν, site], U[μ, siteν⁺])
    Mμν = cmvmul(tmp, tmpϕ)

    tmpϕ = apply_bc(
        apply_bc(ϕ[siteμ⁺ν⁻], bc, site, Val(1), Nμ, Val(μ)),
        bc, site, Val(-1), Nν, Val(ν)
    )
    tmp = cmatmul_od(U[μ, site], U[ν, siteμ⁺ν⁻]) + cmatmul_do(U[ν, siteν⁻], U[μ, siteν⁻])
    Mμν += cmvmul(tmp, tmpϕ)

    tmpϕ = apply_bc(
        apply_bc(ϕ[siteμ⁻ν⁺], bc, site, Val(-1), Nμ, Val(μ)),
        bc, site, Val(1), Nν, Val(ν)
    )
    tmp = cmatmul_do(U[μ, siteμ⁻], U[ν, siteμ⁻]) + cmatmul_od(U[ν, site], U[μ, siteμ⁻ν⁺])
    Mμν += cmvmul(tmp, tmpϕ)

    tmpϕ = apply_bc(
        apply_bc(ϕ[siteμ⁻ν⁻], bc, site, Val(-1), Nμ, Val(μ)),
        bc, site, Val(-1), Nν, Val(ν)
    )
    tmp = cmatmul_dd(U[μ, siteμ⁻], U[ν, siteμ⁻ν⁻]) + cmatmul_dd(U[ν, siteν⁻], U[μ, siteμ⁻ν⁻])
    Mμν += cmvmul(tmp, tmpϕ)
    return im * T(1/4 * staggered_ημν(Val(μ), Val(ν), site)) * Mμν # The extra factor 1/2 is contained in the kernel function
end

"""
    StaggeredTM1DiracOperator(f::AbstractField, mass; bc_str="antiperiodic")
    StaggeredTM1DiracOperator(D::StaggeredDiracOperator, U::Gaugefield)

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
struct StaggeredDiracOperator{B,T,TF,TG,BC} <: AbstractDiracOperator{B,T}
    U::TG
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    mass::Float64
    boundary_condition::BC # Only in time direction
    function StaggeredDiracOperator(
        U::TG, temp::TF, mass, bc::BC
    ) where {B,T,TG<:Gaugefield{B,T},TF<:Spinorfield{B,T},BC}
        return new{B,T,TF,TG,BC}(U, temp, mass, bc)
    end

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
end

function add_gauge_background(
    D::StaggeredDiracOperator{B,T,TF}, U::Gaugefield{B,T}
) where {B,T,TF}
    check_dims(U, D.temp)
    return StaggeredDiracOperator(U, D.temp, D.mass, D.boundary_condition)
end

@inline default_Nf(::StaggeredDiracOperator) = 8
@inline is_staggered(::StaggeredDiracOperator) = true

function solve_dirac!(
    ψ, D::T, ϕ, temps...; tol=1e-14, maxiters=1000
) where {T<:StaggeredDiracOperator}
    return bicg_stab!(ψ, D, ϕ, temps...; tol=tol, maxiters=maxiters)
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
    ψ::TMF, D::StaggeredDiracOperator{CPU,T,TF,TG}, ϕ::TMF
) where {T,TMF<:MultiSpinorfield,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass = T(D.mass)
    bc = D.boundary_condition
    check_dims(ψ, ϕ, U)

    @batch for site in eachindex(ψ)
        for is in 1:num_spinors(ψ)
            ψ[is, site] = staggered_kernel(U, ϕ, site, mass, bc, T, false)
        end
    end

    update_halo!(ψ) # TODO:
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
@inline staggered_η(::Val{5}, site) = ifelse(iseven(site[1] + site[3]), 1, -1)
@inline staggered_ϵμν(::Val{μ}, ::Val{ν}, site) where {μ,ν} =
    ifelse(iseven(site[μ] + site[ν]), 1, -1)

@inline function ξ5(::Type{T}) where {T}
    return @SArray [
        Complex{T}(-1, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(-1, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(1, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(1, 0)
    ]
end

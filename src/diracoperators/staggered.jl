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
        temp = Spinorfield(f; staggered=true, hw=1)
        boundary_condition = create_bc(bc_str, f.topology)
        TG = Nothing
        TF = typeof(temp)
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
    ψ, D::T, ϕ, temps...; tol=1e-14, maxiters=1000, datafile=""
) where {T<:StaggeredDiracOperator}
    D_dagg = Daggered(D)
    return cgnr!(ψ, D, D_dagg, ϕ, temps[1], temps[2], temps[3], temps[4]; tol, maxiters, datafile)
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ψ::TF, D::StaggeredDiracOperator{B,T,TF,TG}, ϕ::TF
) where {B,T,M,TF<:StaggeredSpinorfield{B,T,M},TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass = T(D.mass)
    bc = D.boundary_condition

    parallelfor(eachindex(ψ, ϕ, U), B, Val(M), (U, ϕ), (ψ,), (U, ϕ, ψ)) do site, (U, ϕ, ψ)
        @inbounds ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, T, false)
    end

    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{StaggeredDiracOperator{B,T,TF,TG,BC}}, ϕ::TF
) where {B,T,M,TF<:StaggeredSpinorfield{B,T,M},TG,BC}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass = T(D.parent.mass)
    bc = D.parent.boundary_condition

    parallelfor(eachindex(ψ, ϕ, U), B, Val(M), (U, ϕ), (ψ,), (U, ϕ, ψ)) do site, (U, ϕ, ψ)
        @inbounds ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, T, true)
    end

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

@inline function staggered_kernel(U, ϕ, site, mass, bc, ::Type{T}, dagg::Bool) where {T}
    @inbounds begin
        sgn = dagg ? T(-1) : T(1)
        ψₙ = 2mass * ϕ[site]
        NT = size(U, 4)
        # use @nexprs here to statically generate the loop
        # this makes it so Val(i) is well defined at each iteration and no type-instabilities arise
        @nexprs 4 μ -> (
            Nμ = axes(U, μ);
            siteμ⁺ = move(site, μ, 1, Nμ);
            siteμ⁻ = move(site, μ, -1, Nμ);
            η = sgn * staggered_η(Val(μ), site, T);
            ϕ⁺ = apply_bc(ϕ[siteμ⁺], bc, site, Val(1), NT, Val(μ));
            ϕ⁻ = apply_bc(ϕ[siteμ⁻], bc, site, Val(-1), NT, Val(μ));
            ψₙ += η * (cmvmul(U[μ, site], ϕ⁺) - cmvmul_d(U[μ, siteμ⁻], ϕ⁻))
        )
    end

    return T(0.5) * ψₙ
end

# Use Val to reduce the amount of if-statements in the kernel
@inline staggered_η(::Val{1}, site, ::Type{T}) where {T} = T(1)
@inline staggered_η(::Val{2}, site, ::Type{T}) where {T} = @inbounds ifelse(iseven(site[Int32(1)]), T(1), T(-1))
@inline staggered_η(::Val{3}, site, ::Type{T}) where {T} = @inbounds ifelse(iseven(site[Int32(1)] + site[Int32(2)]), T(1), T(-1))
@inline staggered_η(::Val{4}, site, ::Type{T}) where {T} = @inbounds ifelse(iseven(site[Int32(1)] + site[Int32(2)] + site[Int32(3)]), T(1), T(-1))
@inline staggered_η(::Val{5}, site, ::Type{T}) where {T} = @inbounds ifelse(iseven(site[Int32(1)] + site[Int32(3)]), T(1), T(-1))
@inline staggered_ϵμν(::Val{μ}, ::Val{ν}, site) where {μ,ν} =
    @inbounds ifelse(iseven(site[μ] + site[ν]), 1, -1)

@inline function ξ5(::Type{T}) where {T}
    return SMatrix{4,4,Complex{T},16}(
        Complex{T}(-1, 0), Complex{T}(0, 0), Complex{T}(0, 0), Complex{T}(0, 0),
        Complex{T}(0, 0), Complex{T}(-1, 0), Complex{T}(0, 0), Complex{T}(0, 0),
        Complex{T}(0, 0), Complex{T}(0, 0), Complex{T}(1, 0), Complex{T}(0, 0),
        Complex{T}(0, 0), Complex{T}(0, 0), Complex{T}(0, 0), Complex{T}(1, 0)
    )
end

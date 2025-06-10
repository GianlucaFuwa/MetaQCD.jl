"""
    WilsonDiracOperator(::AbstractField, mass; bc_str="antiperiodic", r=1, csw=0)
    WilsonDiracOperator(D::WilsonDiracOperator, U::Gaugefield)

Create a free Wilson Dirac Operator with mass `mass` and Wilson parameter `r`.

`bc_str` can either be `"periodic"` or `"antiperiodic"` and specifies the boundary
condition in the time direction.

If `csw ≠ 0`, a clover term is included. 

This object cannot be applied to a fermion vector, since it lacks a gauge background.
A Wilson Dirac operator with gauge background is created by applying it to a `Gaugefield`
`U` like `D_gauge = D(U)`

# Type Parameters:
- `B`: Backend (CPU / CUDA / ROCm)
- `T`: Floating point precision
- `TF`: Type of the `Spinorfield` used to store intermediate results when using the 
        Hermitian version of the operator
- `TG`: Type of the underlying `Gaugefield`
- `C`: Boolean declaring whether the operator is clover improved or not
- `BC`: Boundary Condition in time direction
"""
struct WilsonDiracOperator{B,T,C,TF,TG,BC} <: AbstractDiracOperator{B,T}
    U::TG
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    mass::Float64
    κ::Float64
    r::Float64
    csw::Float64
    boundary_condition::BC # Only in time direction
    function WilsonDiracOperator(
        U::TG, temp::TF, mass, κ, r, csw, ::Val{C}, bc::BC
    ) where {B,T,C,TG<:Gaugefield{B,T},TF<:WilsonSpinorfield{B,T},BC}
        return new{B,T,C,TF,TG,BC}(U, temp, mass, κ, r, csw, bc)
    end

    function WilsonDiracOperator(
        f::AbstractField{B,T}, mass; bc_str="antiperiodic", r=1, csw=0, kwargs...
    ) where {B,T}
        @assert r == 1 "Only r=1 in Wilson Dirac supported for now"
        κ = 1 / (2mass + 8)
        U = nothing
        C = csw == 0 ? false : true
        temp = Spinorfield(f)
        TG = Nothing
        TF = typeof(temp)
        boundary_condition = create_bc(bc_str, f.topology)
        BC = typeof(boundary_condition)
        return new{B,T,C,TF,TG,BC}(U, temp, mass, κ, r, csw, boundary_condition)
    end
end

function add_gauge_background(
    D::WilsonDiracOperator{B,T,C,TF}, U::Gaugefield{B,T}
) where {B,T,C,TF}
    check_dims(U, D.temp)
    bc = D.boundary_condition
    return WilsonDiracOperator(U, D.temp, D.mass, D.κ, D.r, D.csw, Val(C), bc)
end

@inline default_Nf(::WilsonDiracOperator) = 2
@inline is_staggered(::WilsonDiracOperator) = false
@inline has_clover_term(::WilsonDiracOperator{B,T,C}) where {B,T,C} = C
@inline has_clover_term(::Daggered{W}) where {B,T,C,W<:WilsonDiracOperator{B,T,C}} = C
@inline has_clover_term(::DdaggerD{W}) where {B,T,C,W<:WilsonDiracOperator{B,T,C}} = C

function solve_dirac!(
    ψ, D::T, ϕ, temps...; tol=1e-16, maxiters=1000
) where {T<:WilsonDiracOperator}
    return bicg_stab!(ψ, D, ϕ, temps...; tol=tol, maxiters=maxiters)
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ψ::TF, D::WilsonDiracOperator{CPU,T,C,TF,TG}, ϕ::TF
) where {T,C,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass_term = T(8 + 2 * D.mass)
    csw = D.csw
    bc = D.boundary_condition

    @batch for site in eachindex(ψ, ϕ, U)
        ψ[site] = wilson_kernel(U, ϕ, site, mass_term, bc, T, Val(1))
    end

    if has_clover_term(D)
        fac = T(-csw / 2)

        @batch for site in eachindex(ψ)
            ψ[site] += clover_kernel(U, ϕ, site, fac, T)
        end
    end

    update_halo!(ψ)
    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{WilsonDiracOperator{CPU,T,C,TF,TG,BC}}, ϕ::TF
) where {T,C,TF,TG,BC}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass_term = T(8 + 2 * D.parent.mass)
    csw = D.parent.csw
    bc = D.parent.boundary_condition

    @batch for site in eachindex(ψ, ϕ, U)
        ψ[site] = wilson_kernel(U, ϕ, site, mass_term, bc, T, Val(-1))
    end

    if has_clover_term(D)
        fac = T(-csw / 2)

        @batch for site in eachindex(ψ, ϕ, U)
            ψ[site] += clover_kernel(U, ϕ, site, fac, T)
        end
    end

    update_halo!(ψ)
    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::DdaggerD{WilsonDiracOperator{B,T,C,TF,TG,BC}}, ϕ::TF
) where {B,T,C,TF,TG,BC}
    temp = D.parent.temp
    mul!(temp, D.parent, ϕ) # temp = Dϕ
    mul!(ψ, adjoint(D.parent), temp) # ψ = D†Dϕ
    return nothing
end

function wilson_kernel(U, ϕ, site, mass_term, bc, ::Type{T}, ::Val{dagg}) where {T,dagg}
    # dagg can be 1 or -1; if it's -1 then we swap (1 - γᵨ) with (1 + γᵨ) and vice versa
    # We have to wrap in a Val for the same reason as in the next comment
    NX, NY, NZ, NT = dims(U)
    ψₙ = mass_term * ϕ[site] # factor 1/2 is included at the end

    # use @nexprs here to statically generate the loop
    # this makes it so Val(i) is well defined at each iteration and no type-instabilities arise
    @nexprs 4 i -> (
        siteμ⁺ = move(site, i, 1, (NX, NY, NZ, NT)[i]);
        siteμ⁻ = move(site, i, -1, (NX, NY, NZ, NT)[i]);
        ϕ⁺ = apply_bc(ϕ[siteμ⁺], bc, site, Val(1), NT, Val(i));
        ϕ⁻ = apply_bc(ϕ[siteμ⁻], bc, site, Val(-1), NT, Val(i));
        ψₙ -= cmvmul_spin_proj(U[i, site], ϕ⁺, Val(-i*dagg), Val(false));
        ψₙ -= cmvmul_spin_proj(U[i, siteμ⁻], ϕ⁻, Val(i*dagg), Val(true))
    )
    return T(0.5) * ψₙ
end

function clover_kernel(U, ϕ, site, fac, ::Type{T}) where {T}
    # Observed that it makes a difference whether we only make F antihermitian or traceless
    # antihermitian in the accuracy of the derivative --> TA makes it worse
    # is most severe when U is unsmeared
    ϕ_n = ϕ[site]
    Cₙₘ = zero(ϕ_n)

    C₁₂ = clover_square(U, 1, 2, site, 1)
    F₁₂ = C₁₂ - C₁₂'
    Cₙₘ += cmvmul_color(F₁₂, σμν_spin_mul(ϕ_n, Val(1), Val(2)))

    C₁₃ = clover_square(U, 1, 3, site, 1)
    F₁₃ = C₁₃ - C₁₃'
    Cₙₘ += cmvmul_color(F₁₃, σμν_spin_mul(ϕ_n, Val(1), Val(3)))

    C₁₄ = clover_square(U, 1, 4, site, 1)
    F₁₄ = C₁₄ - C₁₄'
    Cₙₘ += cmvmul_color(F₁₄, σμν_spin_mul(ϕ_n, Val(1), Val(4)))

    C₂₃ = clover_square(U, 2, 3, site, 1)
    F₂₃ = C₂₃ - C₂₃'
    Cₙₘ += cmvmul_color(F₂₃, σμν_spin_mul(ϕ_n, Val(2), Val(3)))

    C₂₄ = clover_square(U, 2, 4, site, 1)
    F₂₄ = C₂₄ - C₂₄'
    Cₙₘ += cmvmul_color(F₂₄, σμν_spin_mul(ϕ_n, Val(2), Val(4)))

    C₃₄ = clover_square(U, 3, 4, site, 1)
    F₃₄ = C₃₄ - C₃₄'
    Cₙₘ += cmvmul_color(F₃₄, σμν_spin_mul(ϕ_n, Val(3), Val(4)))
    return Complex{T}(fac * im / 8) * Cₙₘ
end

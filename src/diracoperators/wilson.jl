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
struct WilsonDiracOperator{B,T,C,TF,TG,BC,TT} <: AbstractDiracOperator{B,T}
    U::TG
    Fμν::TT
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    mass::Float64
    κ::Float64
    r::Float64
    csw::Float64
    boundary_condition::BC # Only in time direction
    function WilsonDiracOperator(
        U::TG, Fμν::TT, temp::TF, mass, κ, r, csw, ::Val{C}, bc::BC
    ) where {B,T,C,TG<:Gaugefield{B,T},TF<:WilsonSpinorfield{B,T},BC,TT}
        return new{B,T,C,TF,TG,BC,TT}(U, Fμν, temp, mass, κ, r, csw, bc)
    end

    function WilsonDiracOperator(
        f::AbstractField{B,T}, mass; bc_str="antiperiodic", r=1, csw=0, kwargs...
    ) where {B,T}
        @assert r == 1 "Only r=1 in Wilson Dirac supported for now"
        κ = 1 / (2mass + 8)
        U = nothing
        C = csw == 0 ? false : true
        hw = C ? 2 : 1
        Fμν = C ? Tensorfield(f; no_halo=true) : nothing
        temp = Spinorfield(f; hw=hw)
        boundary_condition = create_bc(bc_str, f.topology)
        TG = Nothing
        TF = typeof(temp)
        BC = typeof(boundary_condition)
        TT = typeof(Fμν)
        return new{B,T,C,TF,TG,BC,TT}(U, Fμν, temp, mass, κ, r, csw, boundary_condition)
    end
end

function add_gauge_background(
    D::WilsonDiracOperator{B,T,C,TF}, U::Gaugefield{B,T}
) where {B,T,C,TF}
    check_dims(U, D.temp)
    bc = D.boundary_condition
    C && fieldstrength_eachsite!(Clover(), D.Fμν, U)
    return WilsonDiracOperator(U, D.Fμν, D.temp, D.mass, D.κ, D.r, D.csw, Val(C), bc)
end

@inline default_Nf(::WilsonDiracOperator) = 2
@inline is_staggered(::WilsonDiracOperator) = false
@inline has_clover_term(::WilsonDiracOperator{B,T,C}) where {B,T,C} = C
@inline has_clover_term(::Daggered{W}) where {B,T,C,W<:WilsonDiracOperator{B,T,C}} = C
@inline has_clover_term(::DdaggerD{W}) where {B,T,C,W<:WilsonDiracOperator{B,T,C}} = C

function solve_dirac!(
    ψ, D::T, ϕ, temps...; tol=1e-16, maxiters=1000, datafile=""
) where {T<:WilsonDiracOperator}
    D_dagg = Daggered(D)
    return cgnr!(ψ, D, D_dagg, ϕ, temps[1], temps[2], temps[3], temps[4]; tol, maxiters, datafile)
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ψ::TF, D::WilsonDiracOperator{B,T,C,TF,TG}, ϕ::TF
) where {B,T,M,C,TF<:WilsonSpinorfield{B,T,M},TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    Fμν = D.Fμν
    mass_term = T(8 + 2 * D.mass)
    csw = D.csw
    bc = D.boundary_condition
    fac = T(-csw / 2)

    parallelfor(eachindex(ψ, ϕ, U), B, Val(M), (U, ϕ), (ψ,), (U, ϕ, ψ, Fμν)) do site, (U, ϕ, ψ, Fμν)
        ψ[site] = wilson_kernel(U, Fμν, ϕ, site, mass_term, fac, bc, T, Val(1), Val(C))
    end

    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{WilsonDiracOperator{B,T,C,TF,TG,BC,TT}}, ϕ::TF
) where {B,T,M,C,TF<:WilsonSpinorfield{B,T,M},TG,BC,TT}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    Fμν = D.parent.Fμν
    mass_term = T(8 + 2 * D.parent.mass)
    csw = D.parent.csw
    bc = D.parent.boundary_condition
    fac = T(-csw / 2)

    parallelfor(eachindex(ψ, ϕ, U), B, Val(M), (U, ϕ), (ψ,), (U, ϕ, ψ, Fμν)) do site, (U, ϕ, ψ, Fμν)
        ψ[site] = wilson_kernel(U, Fμν, ϕ, site, mass_term, fac, bc, T, Val(-1), Val(C))
    end

    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::DdaggerD{WilsonDiracOperator{B,T,C,TF,TG,BC,TT}}, ϕ::TF
) where {B,T,C,TF,TG,BC,TT}
    temp = D.parent.temp
    mul!(temp, D.parent, ϕ) # temp = Dϕ
    mul!(ψ, adjoint(D.parent), temp) # ψ = D†Dϕ
    return nothing
end

function wilson_kernel(
    U, ::Any, ϕ, site, mass_term, ::Any, bc, ::Type{T}, ::Val{dagg}, ::Val{false}
) where {T,dagg}
    @inbounds begin
        # dagg can be 1 or -1; if it's -1 then we swap (1 - γᵨ) with (1 + γᵨ) and vice versa
        # We have to wrap in a Val for the same reason as in the next comment
        ϕₙ = ϕ[site]
        ψₙ = mass_term * ϕₙ # factor 1/2 is included at the end
        NT = size(U, 4)
        # use @nexprs here to statically generate the loop
        # this makes it so Val(μ) is well defined at each iteration and no type-instabilities arise
        @nexprs 4 μ -> (
            Nμ = axes(U, μ);
            siteμ⁺ = move(site, μ, 1, Nμ);
            siteμ⁻ = move(site, μ, -1, Nμ);
            ϕ⁺ = apply_bc(ϕ[siteμ⁺], bc, site, Val(1), NT, Val(μ));
            ϕ⁻ = apply_bc(ϕ[siteμ⁻], bc, site, Val(-1), NT, Val(μ));
            ψₙ -= cmvmul_spin_proj(U[μ, site], ϕ⁺, Val(-μ*dagg), Val(false));
            ψₙ -= cmvmul_spin_proj(U[μ, siteμ⁻], ϕ⁻, Val(μ*dagg), Val(true))
        )
    end

    return T(0.5) * ψₙ
end

function wilson_kernel(
    U, Fμν, ϕ, site, mass_term, csw_fac, bc, ::Type{T}, ::Val{dagg}, ::Val{true}
) where {T,dagg}
    @inbounds begin
        # dagg can be 1 or -1; if it's -1 then we swap (1 - γᵨ) with (1 + γᵨ) and vice versa
        # We have to wrap in a Val for the same reason as in the next comment
        ϕₙ = ϕ[site]
        ψₙ = mass_term * ϕₙ # factor 1/2 is included at the end
        NT = size(U, 4)
        # use @nexprs here to statically generate the loop
        # this makes it so Val(μ) is well defined at each iteration and no type-instabilities arise
        @nexprs 4 μ -> (
            Nμ = axes(U, μ);
            siteμ⁺ = move(site, μ, 1, Nμ);
            siteμ⁻ = move(site, μ, -1, Nμ);
            ϕ⁺ = apply_bc(ϕ[siteμ⁺], bc, site, Val(1), NT, Val(μ));
            ϕ⁻ = apply_bc(ϕ[siteμ⁻], bc, site, Val(-1), NT, Val(μ));
            ψₙ -= cmvmul_spin_proj(U[μ, site], ϕ⁺, Val(-μ*dagg), Val(false));
            ψₙ -= cmvmul_spin_proj(U[μ, siteμ⁻], ϕ⁻, Val(μ*dagg), Val(true))
        )
        Cₙ = zero(ϕₙ)
        @nexprs 6 i -> (
            Cₙ += cmvmul_color(Fμν[i, site], σμν_spin_mul(ϕₙ, Val(i)))
        )
    end

    return T(0.5) * ψₙ + T(csw_fac) * Cₙ
end

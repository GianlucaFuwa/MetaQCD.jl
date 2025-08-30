"""
    StaggeredEOPreDiracOperator(f::AbstractField, mass; bc_str="antiperiodic")
    StaggeredEOPreDiracOperator(
        D::Union{StaggeredDiracOperator,StaggeredEOPreDiracOperator},
        U::Gaugefield
    )

Create a free even-odd preconditioned Staggered Dirac Operator with mass `mass`.

`bc_str` can either be `"periodic"` or `"antiperiodic"` and specifies the boundary
condition in the time direction.

using Base: @nexprs
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
struct StaggeredEOPreDiracOperator{B,T,TF,TG,BC} <: AbstractDiracOperator{B,T}
    U::TG
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    mass::Float64
    boundary_condition::BC # Only in time direction
    function StaggeredEOPreDiracOperator(
        U::TG, temp::TF, mass, bc::BC
    ) where {B,T,TG<:Gaugefield{B,T},TF<:StaggeredEOPreSpinorfield{B,T},BC}
        return new{B,T,TF,TG,BC}(U, temp, mass, bc)
    end

    function StaggeredEOPreDiracOperator(
        f::AbstractField{B,T}, mass; bc_str="antiperiodic", kwargs...
    ) where {B,T}
        U = nothing
        temp = even_odd(Spinorfield(f; staggered=true, hw=1))
        boundary_condition = create_bc(bc_str, f.topology)
        TG = Nothing
        TF = typeof(temp)
        BC = typeof(boundary_condition)
        return new{B,T,TF,TG,BC}(U, temp, mass, boundary_condition)
    end
end

function add_gauge_background(
    D::StaggeredEOPreDiracOperator{B,T,TF}, U::Gaugefield{B,T}
) where {B,T,TF}
    check_dims(U, D.temp.parent)
    return StaggeredEOPreDiracOperator(U, D.temp, D.mass, D.boundary_condition)
end

@inline default_Nf(::StaggeredEOPreDiracOperator) = 4
@inline is_staggered(::StaggeredEOPreDiracOperator) = true

function solve_dirac!(
    ψ_eo, D::T, ϕ_eo, temp1, temp2, temp3, temp4, temp5; tol=1e-14, maxiters=1000, datafile=""
) where {T<:StaggeredEOPreDiracOperator}
    error("Not implemented yet")
    # TODO: CGNE
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

    mul_oe!(ψ_eo, U, ϕ_eo, bc, true, false) # ψₒ = Dₒₑϕₑ
    mul_eo!(ψ_eo, U, ψ_eo, bc, false, false) # ψₑ = DₑₒDₒₑϕₑ
    axpby!(mass^2, ϕ_eo, -1, ψ_eo) # ψₑ = m²ϕₑ - DₑₒDₒₑϕₑ
    return nothing
end

function mul_oe!(
    ψ_eo::TF, U::Gaugefield{B,T,M}, ϕ_eo::TF, bc, into_odd, dagg::Bool; fac=1
) where {B,T,M,TF<:SpinorfieldEO{B,T,M}}
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    odd_half = false
    itr = eachindex(odd_half, ψ, ϕ, U)
    padded_bulk = ψ.topology.bulk_sites_padded

    parallelfor(itr, B, Val(M), (U, ϕ_eo), (ψ,), (U, ϕ, ψ)) do o_site, (U, ϕ, ψ)
        site = map_from_half(o_site, padded_bulk)
        _site = into_odd ? o_site : switch_sides(o_site, padded_bulk)
        ψ[_site] = fac * staggered_eo_kernel(U, ϕ, site, bc, T, dagg, padded_bulk)
    end

    return nothing
end

function mul_eo!(
    ψ_eo::TF, U::Gaugefield{B,T,M}, ϕ_eo::TF, bc, into_odd, dagg::Bool; fac=1
) where {B,T,M,TF<:SpinorfieldEO{B,T,M}}
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    even_half = true
    itr = eachindex(even_half, ψ, ϕ, U)
    padded_bulk = ψ.topology.bulk_sites_padded

    parallelfor(itr, B, Val(M), (U, ϕ_eo), (ψ,), (U, ϕ, ψ)) do e_site, (U, ϕ, ψ)
        site = map_from_half(e_site, padded_bulk)
        _site = into_odd ? switch_sides(e_site, padded_bulk) : e_site
        ψ[_site] = fac * staggered_eo_kernel(U, ϕ, site, bc, T, dagg, padded_bulk)
    end

    return nothing
end

@inline function staggered_eo_kernel(
    U, ϕ, site, bc, ::Type{T}, dagg::Bool, padded_bulk
) where {T}
    # sites that begin with a "_" are meant for indexing into the even-odd preconn'ed
    # fermion field 
    sgn = dagg ? -1 : 1
    NT = size(U, 4)
    @inbounds begin
        ψₙ = zero(ϕ[site])

        # use @nexprs here to statically generate the loop
        # this makes it so Val(μ) is well defined at each iteration and no type-instabilities arise
        @nexprs 4 μ -> (
            Nμ = axes(U, μ);
            _siteμ⁺ = map_to_half(move(site, μ, 1, Nμ), padded_bulk);
            siteμ⁻ = move(site, μ, -1, Nμ);
            _siteμ⁻ = map_to_half(siteμ⁻, padded_bulk);
            η = sgn * staggered_η(Val(μ), site, T);
            ψₙ += η * cmvmul(U[μ, site], apply_bc(ϕ[_siteμ⁺], bc, site, Val(1), NT, Val(μ)));
            ψₙ -= η * cmvmul_d(U[μ, siteμ⁻], apply_bc(ϕ[_siteμ⁻], bc, site, Val(-1), NT, Val(μ)))
        )
    end

    return T(0.5) * ψₙ
end

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
        temp = even_odd(Spinorfield(f; staggered=true))
        TG = Nothing
        TF = typeof(temp)
        boundary_condition = create_bc(bc_str, f.topology)
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
    ψ_eo, D::T, ϕ_eo, temp1, temp2, temp3, temp4, temp5; tol=1e-14, maxiters=1000
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
    ψ_eo::TF, U::Gaugefield{CPU,T}, ϕ_eo::TF, bc, into_odd, dagg::Bool; fac=1
) where {T,TF<:SpinorfieldEO{CPU,T}}
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    loc_dims = ψ.topology.local_dims
    loc_dims_padded = ψ.topology.local_dims_padded
    nv = prod(loc_dims)
    origin = ψ.topology.bulk_sites[1]

    #= @batch =# for site in eachindex(:odd, ψ, ϕ, U)
        _site = if into_odd
            eo_site(site, origin, loc_dims..., nv)
        else
            eo_site_switch(site, origin, loc_dims..., nv)
        end
        ψ[_site] = fac * staggered_eo_kernel(
            U, ϕ, site, origin, loc_dims, loc_dims_padded, bc, T, dagg
        )
    end

    update_halo_eo!(ψ)
    return nothing
end

function mul_eo!(
    ψ_eo::TF, U::Gaugefield{CPU,T}, ϕ_eo::TF, bc, into_odd, dagg::Bool; fac=1
) where {T,TF<:SpinorfieldEO{CPU,T}}
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    loc_dims = ψ.topology.local_dims
    loc_dims_padded = ψ.topology.local_dims_padded
    nv = prod(loc_dims)
    origin = ψ.topology.bulk_sites[1]

    #= @batch =# for site in eachindex(:even, ψ, ϕ, U)
        _site = if into_odd
            eo_site_switch(site, origin, loc_dims..., nv)
        else
            eo_site(site, origin, loc_dims..., nv)
        end
        ψ[_site] = fac * staggered_eo_kernel(
            U, ϕ, site, origin, loc_dims, loc_dims_padded, bc, T, dagg
        )
    end

    update_halo_eo!(ψ)
    return nothing
end

function staggered_eo_kernel(
    U, ϕ, site, origin, local_dims, local_dims_padded, bc, ::Type{T}, dagg::Bool
) where {T}
    sgn = dagg ? -1 : 1
    # sites that begin with a "_" are meant for indexing into the even-odd preconn'ed
    # fermion field 
    nx, ny, nz, nt = local_dims
    nv = prod(local_dims)
    NT = local_dims_padded[4]
    ψₙ = zero(ϕ[site])

    # use @nexprs here to statically generate the loop
    # this makes it so Val(i) is well defined at each iteration and no type-instabilities arise
    @nexprs 4 i -> (
        _siteμ⁺ = eo_site(move(site, i, 1, local_dims_padded[i]), origin, nx, ny, nz, nt, nv);
        siteμ⁻ = move(site, i, -1, local_dims_padded[i]);
        _siteμ⁻ = eo_site(siteμ⁻, origin, nx, ny, nz, nt, nv);
        η = sgn * staggered_η(Val(i), site);
        ψₙ += η * cmvmul(U[i, site], apply_bc(ϕ[_siteμ⁺], bc, site, Val(1), NT, Val(i)));
        ψₙ -= η * cmvmul_d(U[i, siteμ⁻], apply_bc(ϕ[_siteμ⁻], bc, site, Val(-1), NT, Val(i)))
    )
    return T(0.5) * ψₙ
end

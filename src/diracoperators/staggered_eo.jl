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

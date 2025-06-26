"""
    StaggeredHoelblingDiracOperator{MT}(f::AbstractField, mass; bc_str="antiperiodic")
    StaggeredHoelblingDiracOperator(D::StaggeredHoelblingDiracOperator, U::Gaugefield)

Create a free Hölbling mass split Staggered Dirac Operator (arXiv:1009.5362) with mass
`mass`.
The type-parameter `MT` determines the kind of operator that is used:
- `MT = 1234`: M12 + M34
- `MT = 1324`: M13 + M24
- `MT = 1342`: M13 + M42

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
struct StaggeredHoelblingDiracOperator{MT,B,T,TF,TG,BC} <: AbstractDiracOperator{B,T}
    U::TG
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    mass::Float64
    c1::Float64
    c2::Float64
    boundary_condition::BC # Only in time direction
    function StaggeredHoelblingDiracOperator{MT}(
        U::TG, temp::TF, mass, c1, c2, bc::BC
    ) where {B,T,MT,TG<:Gaugefield{B,T},TF<:StaggeredSpinorfield{B,T},BC}
        if is_distributed(U)
            @assert U.topology.halo_width >= 2 """
            halo_width must be >= 2 when using hoelbling type staggered fermions
            """
        end
        return new{MT,B,T,TF,TG,BC}(U, temp, mass, c1, c2, bc)
    end

    function StaggeredHoelblingDiracOperator{MT}(
        f::AbstractField{B,T}, mass; bc_str="antiperiodic", kwargs...
    ) where {MT,B,T}
        @assert MT ∈ (1234, 1324, 1342) "Mass term $(MT) not supported"
        U = nothing
        temp = Spinorfield(f; staggered=true, hw=2)
        TG = Nothing
        TF = typeof(temp)
        boundary_condition = create_bc(bc_str, f.topology)
        BC = typeof(boundary_condition)
        return new{MT,B,T,TF,TG,BC}(U, temp, mass, 1.0, 1.0, boundary_condition)
    end
end

function add_gauge_background(
    D::StaggeredHoelblingDiracOperator{MT,B,T,TF,TG,BC}, U::Gaugefield{B,T}
) where {MT,B,T,TF,TG,BC}
    bc = D.boundary_condition
    return StaggeredHoelblingDiracOperator{MT}(U, D.temp, D.mass, D.c1, D.c2, bc)
end

@inline default_Nf(::StaggeredHoelblingDiracOperator) = 2
@inline is_staggered(::StaggeredHoelblingDiracOperator) = true

@inline function get_mass_term(::Val{MT}) where {MT}
    return if MT == 1234
        (Val(1), Val(2), Val(3), Val(4))
    elseif MT == 1324
        (Val(1), Val(3), Val(2), Val(4))
    elseif MT == 1342
        (Val(1), Val(3), Val(4), Val(2))
    end
end

@inline function get_mass_term(::StaggeredHoelblingDiracOperator{MT}) where {MT}
    return get_mass_term(Val(MT))
end

function solve_dirac!(
    ψ, D::T, ϕ, temps...; tol=1e-14, maxiters=1000
) where {T<:StaggeredHoelblingDiracOperator}
    return bicg_stab!(ψ, D, ϕ, temps...; tol=tol, maxiters=maxiters)
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ψ::TF, D::StaggeredHoelblingDiracOperator{MT,B,T,TF,TG}, ϕ::TF
) where {MT,B,T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass = T(D.mass)
    term = get_mass_term(D)
    bc = D.boundary_condition
    # TODO: can hide
    update_halo!(U, ϕ)

    parallelfor(eachindex(ψ, ϕ, U), B) do site
        ψ[site] = staggered_hoelbling_kernel(U, ϕ, site, mass, bc, term, T, false)
    end

    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{StaggeredHoelblingDiracOperator{MT,B,T,TF,TG,BC}}, ϕ::TF
) where {MT,B,T,TF,TG,BC}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass = T(D.parent.mass)
    term = get_mass_term(D.parent)
    bc = D.parent.boundary_condition
    # TODO: can hide
    update_halo!(U, ϕ)

    parallelfor(eachindex(ψ, ϕ, U), B) do site
        ψ[site] = staggered_hoelbling_kernel(U, ϕ, site, mass, bc, term, T, true)
    end

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
    _μ, _ν, _ρ, _σ = term
    NT = size(U, 4)
    ψₙ = (2mass + 4) * ϕ[site] + (
        hoelbling_mass(_μ, _ν, U, ϕ, site, bc, T) +
        hoelbling_mass(_ρ, _σ, U, ϕ, site, bc, T)
    )

    # use @nexprs here to statically generate the loop
    # this makes it so Val(i) is well defined at each iteration and no type-instabilities arise
    @nexprs 4 μ -> (
        Nμ = axes(U, μ);
        siteμ⁺ = move(site, μ, 1, Nμ);
        siteμ⁻ = move(site, μ, -1, Nμ);
        η = sgn * staggered_η(Val(μ), site);
        ϕ⁺ = apply_bc(ϕ[siteμ⁺], bc, site, Val(1), NT, Val(μ));
        ϕ⁻ = apply_bc(ϕ[siteμ⁻], bc, site, Val(-1), NT, Val(μ));
        ψₙ += η * (cmvmul(U[μ, site], ϕ⁺) - cmvmul_d(U[μ, siteμ⁻], ϕ⁻))
    )
    return T(0.5) * ψₙ
end

function hoelbling_mass(::Val{μ}, ::Val{ν}, U, ϕ, site, bc, ::Type{T}) where {μ,ν,T}
    NT = size(U, 4)
    Nμ = axes(U, μ)
    Nν = axes(U, ν)
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteμ⁻ = move(site, μ, -1, Nμ)
    siteν⁺ = move(site, ν, 1, Nν)
    siteν⁻ = move(site, ν, -1, Nν)
    siteμ⁺ν⁺ = move(siteμ⁺, ν, 1, Nν)
    siteμ⁺ν⁻ = move(siteμ⁺, ν, -1, Nν)
    siteμ⁻ν⁺ = move(siteμ⁻, ν, 1, Nν)
    siteμ⁻ν⁻ = move(siteμ⁻, ν, -1, Nν)

    tmpϕ = apply_bc(
        apply_bc(ϕ[siteμ⁺ν⁺], bc, site, Val(1), NT, Val(μ)),
        bc, site, Val(1), NT, Val(ν)
    )
    tmp = cmatmul_oo(U[μ, site], U[ν, siteμ⁺]) + cmatmul_oo(U[ν, site], U[μ, siteν⁺])
    Mμν = cmvmul(tmp, tmpϕ)

    tmpϕ = apply_bc(
        apply_bc(ϕ[siteμ⁺ν⁻], bc, site, Val(1), NT, Val(μ)),
        bc, site, Val(-1), NT, Val(ν)
    )
    tmp = cmatmul_od(U[μ, site], U[ν, siteμ⁺ν⁻]) + cmatmul_do(U[ν, siteν⁻], U[μ, siteν⁻])
    Mμν += cmvmul(tmp, tmpϕ)

    tmpϕ = apply_bc(
        apply_bc(ϕ[siteμ⁻ν⁺], bc, site, Val(-1), NT, Val(μ)),
        bc, site, Val(1), NT, Val(ν)
    )
    tmp = cmatmul_do(U[μ, siteμ⁻], U[ν, siteμ⁻]) + cmatmul_od(U[ν, site], U[μ, siteμ⁻ν⁺])
    Mμν += cmvmul(tmp, tmpϕ)

    tmpϕ = apply_bc(
        apply_bc(ϕ[siteμ⁻ν⁻], bc, site, Val(-1), NT, Val(μ)),
        bc, site, Val(-1), NT, Val(ν)
    )
    tmp = cmatmul_dd(U[μ, siteμ⁻], U[ν, siteμ⁻ν⁻]) + cmatmul_dd(U[ν, siteν⁻], U[μ, siteμ⁻ν⁻])
    Mμν += cmvmul(tmp, tmpϕ)
    return im * T(1/4 * staggered_ημν(Val(μ), Val(ν), site)) * Mμν # The extra factor 1/2 is contained in the kernel function
end

@generated function staggered_ημν(
    ::Val{μ}, ::Val{ν}, site, ::Val{swap}=Val(false)
) where {μ,ν,swap}
    fac1 = (μ < ν) ? 1 : -1
    fac2 = swap ? -1 : 1
    fac = fac1 * fac2

    q_η = if (μ==1 && ν==2) || (μ==2 && ν==1)
        :(return ifelse(iseven(site[2]), 1, -1))
    elseif (μ==1 && ν==3) || (μ==3 && ν==1)
        :(return ifelse(iseven(site[2] + site[3]), 1, -1))
    elseif (μ==1 && ν==4) || (μ==4 && ν==1)
        :(return ifelse(iseven(site[2] + site[3] + site[4]), 1, -1))
    elseif (μ==2 && ν==3) || (μ==3 && ν==2)
        :(return ifelse(iseven(site[3]), 1, -1))
    elseif (μ==2 && ν==4) || (μ==4 && ν==2)
        :(return ifelse(iseven(site[3] + site[4]), 1, -1))
    elseif (μ==3 && ν==4) || (μ==4 && ν==3)
        :(return ifelse(iseven(site[4]), 1, -1))
    end

    q = quote
        $(Expr(:meta, :inline))
        η = $q_η
        return $fac * η
    end

    return q
end


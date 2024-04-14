"""
    StaggeredDiracOperator(U::Gaugefield, mass; anti_periodic=true)

Create an implicit Wilson Dirac Operator with mass `mass`.
If `anti_periodic` is `true` the fermion fields are anti periodic in the time direction.

# Type Parameters:
- `B`: Backend (CPU / CUDA / ROCm)
- `TT`: Type of the `Fermionfield` used to store intermediate results when using the 
        Hermitian version of the operator
- `TG`: Type of the underlying `Gaugefield`
"""
mutable struct StaggeredDiracOperator{B,T,TF,TG} <: AbstractDiracOperator
    mass::Float64
    anti_periodic::Bool # Only in time direction
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    U::TG
    function StaggeredDiracOperator(U::Gaugefield{B,T}, mass, anti_periodic) where {B,T}
        temp = Fermionfield(U; staggered=true)
        return new{B,T,typeof(temp),typeof(U)}(mass, anti_periodic, temp, U)
    end
end

const StaggeredFermionfield{B,T,A} = Fermionfield{B,T,A,1}

struct StaggeredFermionAction{Nf,TD,TT} <: AbstractFermionAction
    D::TD
    temp1::TT # temp for result of cg
    temp2::TT # temp for A*p in cg
    temp3::TT # temp for r in cg
    temp4::TT # temp for p in cg
    function StaggeredFermionAction(U, mass; anti_periodic=true, Nf=4)
        D = StaggeredDiracOperator(U, mass, anti_periodic)
        temp1 = Fermionfield(U; staggered=true)
        temp2 = Fermionfield(temp1)
        temp3 = Fermionfield(temp1)
        temp4 = Fermionfield(temp1)
        return new{Nf,typeof(D),typeof(temp1)}(D, temp1, temp2, temp3, temp4)
    end
end

function solve_D⁻¹x!(
    ϕ, D::T, ψ, temps...; tol=1e-14, maxiters=1000
) where {T<:StaggeredDiracOperator}
    @assert dims(ϕ) == dims(ψ) == dims(D.U)
    bicg_stab!(ϕ, ψ, D, temps[1:5]..., tol=tol, maxiters=maxiters)
    return nothing
end

function calc_fermion_action(
    fermion_action::StaggeredFermionAction, ψ::StaggeredFermionfield
)
    D = fermion_action.D
    DdagD = DdaggerD(D)
    ϕ, temps... = get_cg_temps(fermion_action)
    clear!(ϕ) # initial guess is zero
    solve_D⁻¹x!(ϕ, DdagD, ψ, temps...) # temp1 = (D†D)⁻¹ψ
    return real(dot(ψ, ϕ))
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ϕ::StaggeredFermionfield{CPU,T},
    D::StaggeredDiracOperator{CPU,T},
    ψ::StaggeredFermionfield{CPU,T},
) where {T}
    U = D.U
    mass = T(D.mass)
    anti = D.anti_periodic
    @assert dims(ϕ) == dims(ψ) == dims(U)

    @batch for site in eachindex(ϕ)
        ϕ[site] = staggered_kernel(U, ψ, site, mass, anti, T)
    end

    return nothing
end

function LinearAlgebra.mul!(
    ϕ::StaggeredFermionfield{CPU,T},
    D::Daggered{StaggeredDiracOperator{CPU,T,TG,TF}},
    ψ::StaggeredFermionfield{CPU,T},
) where {T,TG,TF}
    U = D.parent.U
    mass = T(D.parent.mass)
    anti = D.parent.anti_periodic
    @assert dims(ϕ) == dims(ψ) == dims(U)

    @batch for site in eachindex(ϕ)
        ϕ[site] = staggered_kernel(U, ψ, site, mass, anti, T, -1)
    end

    return nothing
end

function LinearAlgebra.mul!(
    ϕ::StaggeredFermionfield{CPU,T},
    D::DdaggerD{StaggeredDiracOperator{CPU,T,TG,TF}},
    ψ::StaggeredFermionfield{CPU,T},
) where {T,TG,TF}
    temp = D.parent.temp
    mul!(temp, adjoint(D.parent), ψ) # temp = D†ψ
    mul!(ϕ, D.parent, temp) # ϕ = DD†ψ
    return nothing
end

@inline function staggered_kernel(U, ψ, site, mass, anti, T, sgn=1)
    NX, NY, NZ, NT = dims(U)
    ϕₙ = 2mass * ψ[site]
    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    siteμ⁺ = move(site, 1, 1, NX)
    siteμ⁻ = move(site, 1, -1, NX)
    η = sgn * staggered_η(Val(1), site)
    ϕₙ += η * (cmvmul(U[1, site], ψ[siteμ⁺]) - cmvmul_d(U[1, siteμ⁻], ψ[siteμ⁻]))

    siteμ⁺ = move(site, 2, 1, NY)
    siteμ⁻ = move(site, 2, -1, NY)
    η = sgn * staggered_η(Val(2), site)
    ϕₙ += η * (cmvmul(U[2, site], ψ[siteμ⁺]) - cmvmul_d(U[2, siteμ⁻], ψ[siteμ⁻]))

    siteμ⁺ = move(site, 3, 1, NZ)
    siteμ⁻ = move(site, 3, -1, NZ)
    η = sgn * staggered_η(Val(3), site)
    ϕₙ += η * (cmvmul(U[3, site], ψ[siteμ⁺]) - cmvmul_d(U[3, siteμ⁻], ψ[siteμ⁻]))

    siteμ⁺ = move(site, 4, 1, NT)
    siteμ⁻ = move(site, 4, -1, NT)
    η = sgn * staggered_η(Val(4), site)
    bc⁺ = boundary_factor(anti, site[4], 1, NT)
    bc⁻ = boundary_factor(anti, site[4], -1, NT)
    ϕₙ +=
        η * (cmvmul(U[4, site], bc⁺ * ψ[siteμ⁺]) - cmvmul_d(U[4, siteμ⁻], bc⁻ * ψ[siteμ⁻]))
    return T(0.5) * ϕₙ
end

# Use Val to reduce the amount of if-statements in the kernel
@inline staggered_η(::Val{1}, site) = 1
@inline staggered_η(::Val{2}, site) = ifelse(iseven(site[1]), 1, -1)
@inline staggered_η(::Val{3}, site) = ifelse(iseven(site[1] + site[2]), 1, -1)
@inline staggered_η(::Val{4}, site) = ifelse(iseven(site[1] + site[2] + site[3]), 1, -1)

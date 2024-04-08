mutable struct WilsonDiracOperator{B,T,TG,TT} <: AbstractDiracOperator
    U::TG
    temp::TT # temp for storage of intermediate result for DdaggerD operator
    mass::Float64
    anti_periodic::Bool # Only in time direction
    function WilsonDiracOperator(U::Gaugefield{B,T}, mass, anti_periodic) where {B,T}
        temp = Fermionfield(U)
        return new{B,T,typeof(U),typeof(temp)}(U, temp, mass, anti_periodic)
    end
end

const WilsonFermionfield{B,T,A} = Fermionfield{B,T,A,4}

struct WilsonFermionAction{Nf,TD,TT} <: AbstractFermionAction
    D::TD
    temp1::TT # temp for result of cg
    temp2::TT # temp for A*p in cg
    temp3::TT # temp for r in cg
    temp4::TT # temp for p in cg
    function WilsonFermionAction(U, mass; anti_periodic=true, Nf=1)
        D = WilsonDiracOperator(U, mass, anti_periodic)
        temp1 = Fermionfield(U)
        temp2 = Fermionfield(temp1)
        temp3 = Fermionfield(temp1)
        temp4 = Fermionfield(temp1)
        return new{Nf,typeof(D),typeof(temp1)}(D, temp1, temp2, temp3, temp4)
    end
end

function solve_D⁻¹x!(
    ϕ, D::T, ψ, temps...; tol=1e-14, maxiters=1000
) where {T<:WilsonDiracOperator}
    @assert dims(ϕ) == dims(ψ) == dims(D.U)
    bicg_stab!(ϕ, ψ, D, temps[1:5]...; tol=tol, maxiters=maxiters)
    return nothing
end

function calc_fermion_action(fermion_action::WilsonFermionAction, ψ::WilsonFermionfield)
    D = fermion_action.D
    DdagD = DdaggerD(D)
    temp1, temp2, temp3, temp4 = get_cg_temps(fermion_action)
    clear!(temp1) # initial guess is zero
    solve_D⁻¹x!(temp1, DdagD, ψ, temp2, temp3, temp4) # temp1 = (D†D)⁻¹ψ
    return real(dot(ψ, temp1))
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ϕ::WilsonFermionfield{CPU,T},
    D::WilsonDiracOperator{CPU,T},
    ψ::WilsonFermionfield{CPU,T},
) where {T}
    U = D.U
    mass_term = T(8 + 2 * D.mass)
    anti = D.anti_periodic
    @assert dims(ϕ) == dims(ψ) == dims(U)

    for site in eachindex(ϕ)
        ϕ[site] = wilson_kernel(U, ψ, site, mass_term, anti, T)
    end

    return nothing
end

function LinearAlgebra.mul!(
    ϕ::WilsonFermionfield{CPU,T},
    D::Daggered{WilsonDiracOperator{CPU,T,TG,TF}},
    ψ::WilsonFermionfield{CPU,T},
) where {T,TG,TF}
    U = D.parent.U
    mass_term = T(8 + 2 * D.parent.mass)
    anti = D.parent.anti_periodic
    @assert dims(ϕ) == dims(ψ) == dims(U)

    for site in eachindex(ϕ)
        ϕ[site] = wilson_kernel_dagg(U, ψ, site, mass_term, anti, T)
    end

    return nothing
end

function LinearAlgebra.mul!(
    ϕ::WilsonFermionfield{CPU,T},
    D::DdaggerD{WilsonDiracOperator{CPU,T,TG,TF}},
    ψ::WilsonFermionfield{CPU,T},
) where {T,TG,TF}
    temp = D.parent.temp
    mul!(temp, Daggered(D.parent), ψ) # temp = D†
    mul!(ϕ, D.parent, temp) # ϕ = DD†ψ
    return nothing
end

function wilson_kernel(U, ψ, site, mass_term, anti, T)
    NX, NY, NZ, NT = dims(U)
    ϕₙ = mass_term * ψ[site] # factor 1/2 is included at the end
    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    siteμ⁺ = move(site, 1, 1, NX)
    siteμ⁻ = move(site, 1, -1, NX)
    ϕₙ += cmvmul_spin_proj(U[1, site], ψ[siteμ⁺], Val(-1), Val(false))
    ϕₙ += cmvmul_spin_proj(U[1, siteμ⁻], ψ[siteμ⁻], Val(1), Val(true))

    siteμ⁺ = move(site, 2, 1, NY)
    siteμ⁻ = move(site, 2, -1, NY)
    ϕₙ += cmvmul_spin_proj(U[2, site], ψ[siteμ⁺], Val(-2), Val(false))
    ϕₙ += cmvmul_spin_proj(U[2, siteμ⁻], ψ[siteμ⁻], Val(2), Val(true))

    siteμ⁺ = move(site, 3, 1, NZ)
    siteμ⁻ = move(site, 3, -1, NZ)
    ϕₙ += cmvmul_spin_proj(U[3, site], ψ[siteμ⁺], Val(-3), Val(false))
    ϕₙ += cmvmul_spin_proj(U[3, siteμ⁻], ψ[siteμ⁻], Val(3), Val(true))

    siteμ⁺ = move(site, 4, 1, NT)
    siteμ⁻ = move(site, 4, -1, NT)
    bc⁺ = boundary_factor(anti, site[4], 1, NT)
    bc⁻ = boundary_factor(anti, site[4], -1, NT)
    ϕₙ += cmvmul_spin_proj(U[4, site], bc⁺ * ψ[siteμ⁺], Val(-4), Val(false))
    ϕₙ += cmvmul_spin_proj(U[4, siteμ⁻], bc⁻ * ψ[siteμ⁻], Val(4), Val(true))
    return T(0.5) * ϕₙ
end

function wilson_kernel_dagg(U, ψ, site, mass_term, anti, T)
    NX, NY, NZ, NT = dims(U)
    ϕₙ = mass_term * ψ[site] # factor 1/2 is included at the end
    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    siteμ⁺ = move(site, 1, 1, NX)
    siteμ⁻ = move(site, 1, -1, NX)
    ϕₙ += cmvmul_spin_proj(U[1, siteμ⁻], ψ[siteμ⁻], Val(-1), Val(true))
    ϕₙ += cmvmul_spin_proj(U[1, site], ψ[siteμ⁺], Val(1), Val(false))

    siteμ⁺ = move(site, 2, 1, NY)
    siteμ⁻ = move(site, 2, -1, NY)
    ϕₙ += cmvmul_spin_proj(U[2, siteμ⁻], ψ[siteμ⁻], Val(-2), Val(true))
    ϕₙ += cmvmul_spin_proj(U[2, site], ψ[siteμ⁺], Val(2), Val(false))

    siteμ⁺ = move(site, 3, 1, NZ)
    siteμ⁻ = move(site, 3, -1, NZ)
    ϕₙ += cmvmul_spin_proj(U[3, siteμ⁻], ψ[siteμ⁻], Val(-3), Val(true))
    ϕₙ += cmvmul_spin_proj(U[3, site], ψ[siteμ⁺], Val(3), Val(false))

    siteμ⁺ = move(site, 4, 1, NT)
    siteμ⁻ = move(site, 4, -1, NT)
    bc⁺ = boundary_factor(anti, site[4], 1, NT)
    bc⁻ = boundary_factor(anti, site[4], -1, NT)
    ϕₙ += cmvmul_spin_proj(U[4, siteμ⁻], bc⁻ * ψ[siteμ⁻], Val(-4), Val(true))
    ϕₙ += cmvmul_spin_proj(U[4, site], bc⁺ * ψ[siteμ⁺], Val(4), Val(false))
    return T(0.5) * ϕₙ
end

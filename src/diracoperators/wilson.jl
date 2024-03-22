mutable struct WilsonDiracOperator{B,T,TG,TT} <: AbstractDiracOperator
    U::TG
    temp::TT
    mass::Float64
    function WilsonDiracOperator(U::Gaugefield{B,T}, mass) where {B,T}
        temp = Fermionfield(U)
        return new{B,T,typeof(U),typeof(temp)}(U, temp, mass)
    end
end

const WilsonFermionfield{B,T,A} = Fermionfield{B,T,A,4}

struct WilsonFermionAction{TD,TT} <: AbstractFermionAction
    D::TD
    temp1::TT
    temp2::TT
    temp3::TT
    function WilsonFermionAction(U, mass)
        D = WilsonDiracOperator(U, mass)
        temp1 = Fermionfield(U)
        temp2 = Fermionfield(temp1)
        temp3 = Fermionfield(temp1)
        return new{typeof(D),typeof(temp1)}(D, temp1, temp2, temp3)
    end
end

function calc_fermion_action(fermion_action::WilsonFermionAction, ψ::StaggeredFermionfield)
    D = fermion_action.D
    DdagD = Hermitian(D)
    temp1, temp2, temp3 = get_cg_temps(fermion_action)
    solve_D⁻¹x!(temp1, DdagD, ψ, temp2, temp3)
    mul!(temp2, D, ψ)
    return real(dot(ψ, temp2))
end

function LinearAlgebra.mul!(
    ϕ::WilsonFermionfield{CPU,T},
    D::WilsonDiracOperator{CPU,T},
    ψ::WilsonFermionfield{CPU,T},
) where {T}
    U = D.U
    mass = D.mass
    @assert dims(ϕ) == dims(ψ) == dims(U)

    @batch for site in eachindex(ϕ)
        ϕ[site] = wilson_kernel(U, ψ, site, mass, T)
    end

    return nothing
end

function LinearAlgebra.mul!(
    ϕ::WilsonFermionfield{CPU,T},
    D::Daggered{WilsonDiracOperator{CPU,T,TG,TF}},
    ψ::WilsonFermionfield{CPU,T},
) where {T,TG,TF}
    U = D.parent.U
    mass = D.parent.mass
    @assert dims(ϕ) == dims(ψ) == dims(U)

    @batch for site in eachindex(ϕ)
        ϕ[site] = wilson_kernel_dagg(U, ψ, site, mass, T)
    end

    return nothing
end

function LinearAlgebra.mul!(
    ϕ::WilsonFermionfield{CPU,T},
    D::Hermitian{WilsonDiracOperator{CPU,T,TG,TF}},
    ψ::WilsonFermionfield{CPU,T},
) where {T,TG,TF}
    temp = D.parent.temp_F
    mul!(temp, Daggered(D.parent), ψ) # temp = D†
    mul!(ϕ, D.parent, temp) # ϕ = DD†ψ
    return nothing
end

@inline function wilson_kernel(U, ψ, site, mass, T)
    NX, NY, NZ, NT = dims(U)
    ϕₙ = T(8 + 2mass) * ψ[site] # factor 1/2 is included at the end
    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    siteμ⁺ = move(site, 1, 1, NX)
    siteμ⁻ = move(site, 1, -1, NX)
    ϕₙ += cmvmul_spin_proj(U[1, site], ψ[siteμ⁺], Val(-1))
    ϕₙ += cmvmul_spin_proj(U[1, siteμ⁻], ψ[siteμ⁻], Val(1), Val(true))
    siteμ⁺ = move(site, 2, 1, NY)
    siteμ⁻ = move(site, 2, -1, NY)
    ϕₙ += cmvmul_spin_proj(U[2, site], ψ[siteμ⁺], Val(-2))
    ϕₙ += cmvmul_spin_proj(U[2, siteμ⁻], ψ[siteμ⁻], Val(2), Val(true))
    siteμ⁺ = move(site, 3, 1, NZ)
    siteμ⁻ = move(site, 3, -1, NZ)
    ϕₙ += cmvmul_spin_proj(U[3, site], ψ[siteμ⁺], Val(-3))
    ϕₙ += cmvmul_spin_proj(U[3, siteμ⁻], ψ[siteμ⁻], Val(3), Val(true))
    siteμ⁺ = move(site, 4, 1, NT)
    siteμ⁻ = move(site, 4, -1, NT)
    ϕₙ += cmvmul_spin_proj(U[4, site], ψ[siteμ⁺], Val(-4))
    ϕₙ += cmvmul_spin_proj(U[4, siteμ⁻], ψ[siteμ⁻], Val(4), Val(true))
    return T(0.5) * ϕₙ
end

@inline function wilson_kernel_dagg(U, ψ, site, mass, T)
    NX, NY, NZ, NT = dims(U)
    ϕₙ = T(8 + 2mass) * ψ[site] # factor 1/2 is included at the end
    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    siteμ⁺ = move(site, 1, 1, NX)
    siteμ⁻ = move(site, 1, -1, NX)
    ϕₙ += cmvmul_spin_proj(U[1, siteμ⁻], ψ[siteμ⁻], Val(-1), Val(true))
    ϕₙ += cmvmul_spin_proj(U[1, site], ψ[siteμ⁺], Val(1))
    siteμ⁺ = move(site, 2, 1, NY)
    siteμ⁻ = move(site, 2, -1, NY)
    ϕₙ += cmvmul_spin_proj(U[2, siteμ⁻], ψ[siteμ⁻], Val(-2), Val(true))
    ϕₙ += cmvmul_spin_proj(U[2, site], ψ[siteμ⁺], Val(2))
    siteμ⁺ = move(site, 3, 1, NZ)
    siteμ⁻ = move(site, 3, -1, NZ)
    ϕₙ += cmvmul_spin_proj(U[3, siteμ⁻], ψ[siteμ⁻], Val(-3), Val(true))
    ϕₙ += cmvmul_spin_proj(U[3, site], ψ[siteμ⁺], Val(3))
    siteμ⁺ = move(site, 4, 1, NT)
    siteμ⁻ = move(site, 4, -1, NT)
    ϕₙ += cmvmul_spin_proj(U[4, siteμ⁻], ψ[siteμ⁻], Val(-4), Val(true))
    ϕₙ += cmvmul_spin_proj(U[4, site], ψ[siteμ⁺], Val(4))
    return T(0.5) * ϕₙ
end

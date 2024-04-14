"""
    WilsonDiracOperator(U::Gaugefield, mass; anti_periodic=true, r=1, csw=0)

Create an implicit Wilson Dirac Operator with mass `mass` and Wilson parameter `r`.
If `anti_periodic` is `true` the fermion fields are anti periodic in the time direction.
If `csw ≠ 0`, a clover term is included. 

# Type Parameters:
- `B`: Backend (CPU / CUDA / ROCm)
- `TF`: Type of the `Fermionfield` used to store intermediate results when using the 
        Hermitian version of the operator
- `TG`: Type of the underlying `Gaugefield`
"""
mutable struct WilsonDiracOperator{B,T,TF,TG} <: AbstractDiracOperator
    U::TG
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    mass::Float64
    r::Float64
    csw::Union{Nothing,Float64}
    anti_periodic::Bool # Only in time direction
    function WilsonDiracOperator(
        U::Gaugefield{B,T}, mass; anti_periodic=true, r=1, csw=nothing,
    ) where {B,T}
        @assert r === 1 "Only r=1 in Wilson Dirac supported for now"
        temp = Fermionfield(U)
        return new{B,T,typeof(temp),typeof(U)}(U, temp, mass, r, csw, anti_periodic)
    end
end

const WilsonFermionfield{B,T,A} = Fermionfield{B,T,A,4}

struct WilsonFermionAction{Nf,TD,TF} <: AbstractFermionAction
    D::TD
    temp1::TF # temp for result of cg
    temp2::TF # temp for A*p in cg
    temp3::TF # temp for r in cg
    temp4::TF # temp for p in cg
    function WilsonFermionAction(U, mass; anti_periodic=true, Nf=1)
        D = WilsonDiracOperator(U, mass; anti_periodic=anti_periodic)
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
    csw = D.csw
    anti = D.anti_periodic
    @assert dims(ϕ) == dims(ψ) == dims(U)

    @batch for site in eachindex(ϕ)
        ϕ[site] = wilson_kernel(U, ψ, site, mass_term, anti, T)
    end

    if csw !== nothing
        fac = T(-csw/2)
        @batch for site in eachindex(ϕ)
            ϕ[site] += clover_kernel(U, ψ, site, fac, T)
        end
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
    csw = D.parent.csw
    anti = D.parent.anti_periodic
    @assert dims(ϕ) == dims(ψ) == dims(U)

    @batch for site in eachindex(ϕ)
        ϕ[site] = wilson_kernel_dagg(U, ψ, site, mass_term, anti, T)
    end

    if csw !== nothing
        fac = T(-csw/2)
        @batch for site in eachindex(ϕ)
            ϕ[site] += clover_kernel(U, ψ, site, fac, T)
        end
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
    siteμ⁺ = move(site, 1i32, 1i32, NX)
    siteμ⁻ = move(site, 1i32, -1i32, NX)
    ϕₙ += cmvmul_spin_proj(U[1i32, siteμ⁻], ψ[siteμ⁻], Val(-1), Val(true))
    ϕₙ += cmvmul_spin_proj(U[1i32, site], ψ[siteμ⁺], Val(1), Val(false))

    siteμ⁺ = move(site, 2i32, 1i32, NY)
    siteμ⁻ = move(site, 2i32, -1i32, NY)
    ϕₙ += cmvmul_spin_proj(U[2, siteμ⁻], ψ[siteμ⁻], Val(-2), Val(true))
    ϕₙ += cmvmul_spin_proj(U[2, site], ψ[siteμ⁺], Val(2), Val(false))

    siteμ⁺ = move(site, 3i32, 1i32, NZ)
    siteμ⁻ = move(site, 3i32, -1i32, NZ)
    ϕₙ += cmvmul_spin_proj(U[3i32, siteμ⁻], ψ[siteμ⁻], Val(-3), Val(true))
    ϕₙ += cmvmul_spin_proj(U[3i32, site], ψ[siteμ⁺], Val(3), Val(false))

    siteμ⁺ = move(site, 4i32, 1i32, NT)
    siteμ⁻ = move(site, 4i32, -1i32, NT)
    bc⁺ = boundary_factor(anti, site[4i32], 1i32, NT)
    bc⁻ = boundary_factor(anti, site[4i32], -1i32, NT)
    ϕₙ += cmvmul_spin_proj(U[4i32, siteμ⁻], bc⁻ * ψ[siteμ⁻], Val(-4), Val(true))
    ϕₙ += cmvmul_spin_proj(U[4i32, site], bc⁺ * ψ[siteμ⁺], Val(4), Val(false))
    return T(0.5) * ϕₙ
end

function clover_kernel(U, ψ, site, fac, T)
    Cₙₘ = zero(ψ[site])

    C₁₂ = clover_square(U, 1i32, 2i32, site, 1i32)
    F₁₂ = traceless_antihermitian(C₁₂)
    Cₙₘ += cmvmul(ckron(σ₁₂(T), F₁₂), ψ[site])

    C₁₃ = clover_square(U, 1i32, 3i32, site, 1i32)
    F₁₃ = traceless_antihermitian(C₁₃)
    Cₙₘ += cmvmul(ckron(σ₁₃(T), F₁₃), ψ[site])

    C₁₄ = clover_square(U, 1i32, 4i32, site, 1i32)
    F₁₄ = traceless_antihermitian(C₁₄)
    Cₙₘ += cmvmul(ckron(σ₁₄(T), F₁₄), ψ[site])

    C₂₃ = clover_square(U, 2i32, 3i32, site, 1i32)
    F₂₃ = traceless_antihermitian(C₂₃)
    Cₙₘ += cmvmul(ckron(σ₂₃(T), F₂₃), ψ[site])

    C₂₄ = clover_square(U, 2i32, 4i32, site, 1i32)
    F₂₄ = traceless_antihermitian(C₂₄)
    Cₙₘ += cmvmul(ckron(σ₂₄(T), F₂₄), ψ[site])

    C₃₄ = clover_square(U, 3i32, 4i32, site, 1i32)
    F₃₄ = traceless_antihermitian(C₃₄)
    Cₙₘ += cmvmul(ckron(σ₃₄(T), F₃₄), ψ[site])
    return fac * T(1/4) * Cₙₘ
end

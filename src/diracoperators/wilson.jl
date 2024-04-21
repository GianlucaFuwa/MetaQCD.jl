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
    κ::Float64
    r::Float64
    csw::Union{Nothing,Float64}
    anti_periodic::Bool # Only in time direction
    function WilsonDiracOperator(
        U::Gaugefield{B,T}, mass; anti_periodic=true, r=1, csw=nothing
    ) where {B,T}
        @assert r === 1 "Only r=1 in Wilson Dirac supported for now"
        κ = 1 / (2mass + 8)
        temp = Fermionfield(U)
        return new{B,T,typeof(temp),typeof(U)}(U, temp, mass, κ, r, csw, anti_periodic)
    end
end

const WilsonFermionfield{B,T,A} = Fermionfield{B,T,A,4}

struct WilsonFermionAction{Nf,TD,TF} <: AbstractFermionAction
    D::TD
    cg_x::TF
    cg_temps::Vector{TF}
    rhmc_info::Union{Nothing,RHMCParams}
    rhmc_temps::Union{Nothing,Vector{TF}}
    function WilsonFermionAction(
        U, mass; anti_periodic=true, r=1, csw=nothing, Nf=2, rhmc_order=10, rhmc_prec=42
    )
        D = WilsonDiracOperator(U, mass; anti_periodic=anti_periodic, r=r, csw=csw)
        TD = typeof(D)
        cg_x = Fermionfield(U)
        cg_temps = [Fermionfield(cg_x) for _ in 1:3]
        TF = typeof(cg_x)

        if Nf == 2
            rhmc_info = nothing
            rhmc_temps = nothing
        else
            @assert 4 > Nf > 0 "Nf should be between 1 and 4 (was $Nf)"
            power = Nf//2
            rhmc_info = RHMCParams(power; n=rhmc_order, precision=rhmc_prec)
            rhmc_temps = [Fermionfield(U) for _ in 1:2rhmc_order]
        end
        return new{Nf,TD,TF}(D, cg_x, cg_temps, rhmc_info, rhmc_temps)
    end
end

function solve_D⁻¹x!(
    ψ, D::T, ϕ, temps...; tol=1e-16, maxiters=1000
) where {T<:WilsonDiracOperator}
    @assert dims(ψ) == dims(ϕ) == dims(D.U)
    bicg_stab!(ψ, ϕ, D, temps...; tol=tol, maxiters=maxiters)
    return nothing
end

function calc_fermion_action(fermion_action::WilsonFermionAction{2}, ϕ::WilsonFermionfield)
    D = fermion_action.D
    DdagD = DdaggerD(D)
    ψ = fermion_action.cg_x
    temps = fermion_action.cg_temps

    clear!(ψ) # initial guess is zero
    solve_D⁻¹x!(ψ, DdagD, ϕ, temps...) # temp1 = (D†D)⁻¹ϕ
    Sf = dot(ϕ, ψ)
    return real(Sf)
end

function calc_fermion_action(
    fermion_action::WilsonFermionAction{Nf}, ϕ::WilsonFermionfield
) where {Nf}
    D = fermion_action.D
    DdagD = DdaggerD(D)
    ψ = fermion_action.cg_x
    temps, p = fermion_action.cg_temps
    rhmc = fermion_action.rhmc_info
    n = rhmc.coeffs.n
    rhmc_ψ = ntuple(i -> fermion_action.rhmc_temps[i], n)
    rhmc_p = ntuple(i -> fermion_action.rhmc_temps[n+i], n)

    ψs = (ψ, rhmc_ψ...)
    ps = (p, rhmc_p...)

    for v in ψs
        clear!(v)
    end

    shifts = rhmc.coeffs_inverse.β
    coeffs = rhmc.coeffs_inverse.α
    α₀ = rhmc.coeffs_inverse.α0
    mscg!(ψs, shifts, DdagD, ϕ, temps..., ps)
    clear!(ψ) # D⁻¹ϕ doesn't appear in the partial fraction decomp so we can use it to sum

    axpy!(α₀, ϕ, ψ)
    for i in 1:n
        axpy!(coeffs[i], rhmc_ψ[i], ψ)
    end

    Sf = dot(ϕ, ψ)
    return real(Sf)
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ψ::WilsonFermionfield{CPU,T},
    D::WilsonDiracOperator{CPU,T},
    ϕ::WilsonFermionfield{CPU,T},
) where {T}
    U = D.U
    mass_term = T(8 + 2 * D.mass)
    csw = D.csw
    anti = D.anti_periodic
    @assert dims(ψ) == dims(ϕ) == dims(U)

    @batch for site in eachindex(ψ)
        ψ[site] = wilson_kernel(U, ϕ, site, mass_term, anti, T)
    end

    if csw !== nothing
        fac = T(-csw / 2)
        @batch for site in eachindex(ψ)
            ψ[site] += clover_kernel(U, ϕ, site, fac, T)
        end
    end

    return nothing
end

function LinearAlgebra.mul!(
    ψ::WilsonFermionfield{CPU,T},
    D::Daggered{WilsonDiracOperator{CPU,T,TG,TF}},
    ϕ::WilsonFermionfield{CPU,T},
) where {T,TG,TF}
    U = D.parent.U
    mass_term = T(8 + 2 * D.parent.mass)
    csw = D.parent.csw
    anti = D.parent.anti_periodic
    @assert dims(ψ) == dims(ϕ) == dims(U)

    @batch for site in eachindex(ψ)
        ψ[site] = wilson_kernel(U, ϕ, site, mass_term, anti, T, Val(-1))
    end

    if csw !== nothing
        fac = T(-csw / 2)
        @batch for site in eachindex(ψ)
            ψ[site] += clover_kernel(U, ϕ, site, fac, T)
        end
    end

    return nothing
end

function LinearAlgebra.mul!(
    ψ::WilsonFermionfield{CPU,T},
    D::DdaggerD{WilsonDiracOperator{CPU,T,TG,TF}},
    ϕ::WilsonFermionfield{CPU,T},
) where {T,TG,TF}
    temp = D.parent.temp
    mul!(temp, Daggered(D.parent), ϕ) # temp = D†
    mul!(ψ, D.parent, temp) # ψ = DD†ϕ
    return nothing
end

function wilson_kernel(U, ϕ, site, mass_term, anti, T, ::Val{dagg}=Val(1)) where {dagg}
    # dagg can be 1 or -1; if it's -1 then we swap (1 - γᵨ) with (1 + γᵨ) and vice versa
    # We have to wrap in a Val for the same reason as in the next comment
    NX, NY, NZ, NT = dims(U)
    ψₙ = mass_term * ϕ[site] # factor 1/2 is included at the end
    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    siteμ⁺ = move(site, 1i32, 1i32, NX)
    siteμ⁻ = move(site, 1i32, -1i32, NX)
    ψₙ -= cmvmul_spin_proj(U[1i32, site], ϕ[siteμ⁺], Val(-1dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[1i32, siteμ⁻], ϕ[siteμ⁻], Val(1dagg), Val(true))

    siteμ⁺ = move(site, 2i32, 1i32, NY)
    siteμ⁻ = move(site, 2i32, -1i32, NY)
    ψₙ -= cmvmul_spin_proj(U[2i32, site], ϕ[siteμ⁺], Val(-2dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[2i32, siteμ⁻], ϕ[siteμ⁻], Val(2dagg), Val(true))

    siteμ⁺ = move(site, 3i32, 1i32, NZ)
    siteμ⁻ = move(site, 3i32, -1i32, NZ)
    ψₙ -= cmvmul_spin_proj(U[3i32, site], ϕ[siteμ⁺], Val(-3dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[3i32, siteμ⁻], ϕ[siteμ⁻], Val(3dagg), Val(true))

    siteμ⁺ = move(site, 4i32, 1i32, NT)
    siteμ⁻ = move(site, 4i32, -1i32, NT)
    bc⁺ = boundary_factor(anti, site[4i32], 1i32, NT)
    bc⁻ = boundary_factor(anti, site[4i32], -1i32, NT)
    ψₙ -= cmvmul_spin_proj(U[4i32, site], bc⁺ * ϕ[siteμ⁺], Val(-4dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[4i32, siteμ⁻], bc⁻ * ϕ[siteμ⁻], Val(4dagg), Val(true))
    return T(0.5) * ψₙ
end

function clover_kernel(U, ϕ, site, fac, T)
    Cₙₘ = zero(ϕ[site])

    C₁₂ = clover_square(U, 1i32, 2i32, site, 1i32)
    F₁₂ = traceless_antihermitian(C₁₂)
    Cₙₘ += cmvmul(ckron(σ₁₂(T), F₁₂), ϕ[site])

    C₁₃ = clover_square(U, 1i32, 3i32, site, 1i32)
    F₁₃ = traceless_antihermitian(C₁₃)
    Cₙₘ += cmvmul(ckron(σ₁₃(T), F₁₃), ϕ[site])

    C₁₄ = clover_square(U, 1i32, 4i32, site, 1i32)
    F₁₄ = traceless_antihermitian(C₁₄)
    Cₙₘ += cmvmul(ckron(σ₁₄(T), F₁₄), ϕ[site])

    C₂₃ = clover_square(U, 2i32, 3i32, site, 1i32)
    F₂₃ = traceless_antihermitian(C₂₃)
    Cₙₘ += cmvmul(ckron(σ₂₃(T), F₂₃), ϕ[site])

    C₂₄ = clover_square(U, 2i32, 4i32, site, 1i32)
    F₂₄ = traceless_antihermitian(C₂₄)
    Cₙₘ += cmvmul(ckron(σ₂₄(T), F₂₄), ϕ[site])

    C₃₄ = clover_square(U, 3i32, 4i32, site, 1i32)
    F₃₄ = traceless_antihermitian(C₃₄)
    Cₙₘ += cmvmul(ckron(σ₃₄(T), F₃₄), ϕ[site])
    return (fac * im * T(1 / 4)) * Cₙₘ
end

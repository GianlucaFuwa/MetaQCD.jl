"""
    StaggeredDiracOperator(U::Gaugefield, mass; anti_periodic=true)

Create an implicit Wilson Dirac Operator with mass `mass`.
If `anti_periodic` is `true` the fermion fields are anti periodic in the time direction.

# Type Parameters:
- `B`: Backend (CPU / CUDA / ROCm)
- `TF`: Type of the `Fermionfield` used to store intermediate results when using the 
        Hermitian version of the operator
- `TG`: Type of the underlying `Gaugefield`
"""
mutable struct StaggeredDiracOperator{B,T,TF,TG} <: AbstractDiracOperator
    mass::Float64
    anti_periodic::Bool # Only in time direction
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    U::TG
    function StaggeredDiracOperator(
        U::Gaugefield{B,T}, mass; anti_periodic=anti_periodic
    ) where {B,T}
        temp = Fermionfield(U; staggered=true)
        return new{B,T,typeof(temp),typeof(U)}(mass, anti_periodic, temp, U)
    end
end

const StaggeredFermionfield{B,T,A} = Fermionfield{B,T,A,1}

struct StaggeredFermionAction{Nf,TD,TF} <: AbstractFermionAction
    D::TD
    cg_x::TF
    cg_temps::Vector{TF}
    rhmc_info::Union{Nothing,RHMCParams}
    rhmc_temps::Union{Nothing,Vector{TF}}
    tol::Float64
    function StaggeredFermionAction(
        U, mass; anti_periodic=true, Nf=8, rhmc_order=10, rhmc_prec=42, tol=1e-14
    )
        D = StaggeredDiracOperator(U, mass; anti_periodic=anti_periodic)
        TD = typeof(D)
        cg_x = Fermionfield(U; staggered=true)
        cg_temps = [Fermionfield(cg_x) for _ in 1:3]
        TF = typeof(cg_x)

        if Nf == 8
            rhmc_info = nothing
            rhmc_temps = nothing
        else
            @assert 4 > Nf > 0 "Nf should be between 0 and 4 (was $Nf)"
            power = Nf//8
            rhmc_info = RHMCParams(power; n=rhmc_order, precision=rhmc_prec)
            rhmc_temps = [Fermionfield(U; staggered=true) for _ in 1:2rhmc_order]
        end
        return new{Nf,TD,TF}(D, cg_x, cg_temps, rhmc_info, rhmc_temps, tol)
    end
end

function solve_D⁻¹x!(
    ψ, D::T, ϕ, temps...; tol=1e-14, maxiters=1000
) where {T<:StaggeredDiracOperator}
    @assert dims(ψ) == dims(ϕ) == dims(D.U)
    bicg_stab!(ψ, ϕ, D, temps...; tol=tol, maxiters=maxiters)
    return nothing
end

function calc_fermion_action(
    fermion_action::StaggeredFermionAction{8}, ϕ::StaggeredFermionfield
)
    D = fermion_action.D
    DdagD = DdaggerD(D)
    ψ = fermion_action.cg_x
    temps = fermion_action.cg_temps

    clear!(ψ) # initial guess is zero
    solve_D⁻¹x!(ψ, DdagD, ϕ, temps...; tol=fermion_action.tol) # ψ = (D†D)⁻¹ϕ
    Sf = dot(ϕ, ψ)
    return real(Sf)
end

function calc_fermion_action(
    fermion_action::StaggeredFermionAction{Nf}, ϕ::StaggeredFermionfield
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
    mscg!(ψs, shifts, DdagD, ϕ, temps..., ps; tol=fermion_action.tol)
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
    ψ::StaggeredFermionfield{CPU,T},
    D::StaggeredDiracOperator{CPU,T},
    ϕ::StaggeredFermionfield{CPU,T},
) where {T}
    U = D.U
    mass = T(D.mass)
    anti = D.anti_periodic
    @assert dims(ψ) == dims(ϕ) == dims(U)

    @batch for site in eachindex(ψ)
        ψ[site] = staggered_kernel(U, ϕ, site, mass, anti, T)
    end

    return nothing
end

function LinearAlgebra.mul!(
    ψ::StaggeredFermionfield{CPU,T},
    D::Daggered{StaggeredDiracOperator{CPU,T,TG,TF}},
    ϕ::StaggeredFermionfield{CPU,T},
) where {T,TG,TF}
    U = D.parent.U
    mass = T(D.parent.mass)
    anti = D.parent.anti_periodic
    @assert dims(ψ) == dims(ϕ) == dims(U)

    @batch for site in eachindex(ψ)
        ψ[site] = staggered_kernel(U, ϕ, site, mass, anti, T, -1)
    end

    return nothing
end

function LinearAlgebra.mul!(
    ψ::StaggeredFermionfield{CPU,T},
    D::DdaggerD{StaggeredDiracOperator{CPU,T,TG,TF}},
    ϕ::StaggeredFermionfield{CPU,T},
) where {T,TG,TF}
    temp = D.parent.temp
    mul!(temp, adjoint(D.parent), ϕ) # temp = D†ϕ
    mul!(ψ, D.parent, temp) # ψ = DD†ϕ
    return nothing
end

function staggered_kernel(U, ϕ, site, mass, anti, T, sgn=1)
    NX, NY, NZ, NT = dims(U)
    ψₙ = 2mass * ϕ[site]
    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    siteμ⁺ = move(site, 1, 1, NX)
    siteμ⁻ = move(site, 1, -1, NX)
    η = sgn * staggered_η(Val(1), site)
    ψₙ += η * cmvmul(U[1, site], ϕ[siteμ⁺])
    ψₙ -= η * cmvmul_d(U[1, siteμ⁻], ϕ[siteμ⁻])

    siteμ⁺ = move(site, 2, 1, NY)
    siteμ⁻ = move(site, 2, -1, NY)
    η = sgn * staggered_η(Val(2), site)
    ψₙ += η * cmvmul(U[2, site], ϕ[siteμ⁺])
    ψₙ -= η * cmvmul_d(U[2, siteμ⁻], ϕ[siteμ⁻])

    siteμ⁺ = move(site, 3, 1, NZ)
    siteμ⁻ = move(site, 3, -1, NZ)
    η = sgn * staggered_η(Val(3), site)
    ψₙ += η * cmvmul(U[3, site], ϕ[siteμ⁺])
    ψₙ -= η * cmvmul_d(U[3, siteμ⁻], ϕ[siteμ⁻])

    siteμ⁺ = move(site, 4, 1, NT)
    siteμ⁻ = move(site, 4, -1, NT)
    bc⁺ = boundary_factor(anti, site[4], 1, NT)
    bc⁻ = boundary_factor(anti, site[4], -1, NT)
    η = sgn * staggered_η(Val(4), site)
    ψₙ += η * cmvmul(U[4, site], bc⁺ * ϕ[siteμ⁺])
    ψₙ -= η * cmvmul_d(U[4, siteμ⁻], bc⁻ * ϕ[siteμ⁻])
    return T(0.5) * ψₙ
end

# Use Val to reduce the amount of if-statements in the kernel
@inline staggered_η(::Val{1}, site) = 1
@inline staggered_η(::Val{2}, site) = ifelse(iseven(site[1]), 1, -1)
@inline staggered_η(::Val{3}, site) = ifelse(iseven(site[1] + site[2]), 1, -1)
@inline staggered_η(::Val{4}, site) = ifelse(iseven(site[1] + site[2] + site[3]), 1, -1)

mutable struct StaggeredDiracOperator{B,T,TG,TT} <: AbstractDiracOperator
    U::TG
    temp::TT
    mass::Float64
    function StaggeredDiracOperator(U::Gaugefield{B,T}, mass) where {B,T}
        temp = Fermionfield(U, true)
        return new{B,T,typeof(U),typeof(temp)}(U, temp, mass)
    end
end

const StaggeredFermionfield{B,T,A} = Fermionfield{B,T,A,1}

struct StaggeredFermionAction{TD,TT} <: AbstractFermionAction
    D::TD
    temp1::TT
    temp2::TT
    function StaggeredFermionAction(U, mass)
        D = StaggeredDiracOperator(U, mass)
        temp1 = Fermionfield(U, true)
        temp2 = Fermionfield(U, true)
        return new{typeof(D)}(D, temp1, temp2)
    end
end

function LinearAlgebra.mul!(
    ϕ::StaggeredFermionfield{CPU,T},
    D::StaggeredDiracOperator{CPU,T},
    ψ::StaggeredFermionfield{CPU,T},
) where {T}
    U = D.U
    mass = T(D.mass)
    @assert dims(ϕ) == dims(ψ) == dims(U)

    @batch for site in eachindex(ϕ)
        ϕ[1, site] = staggered_kernel(U, ψ, site, mass, T)
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
    @assert dims(ϕ) == dims(ψ) == dims(U)

    @batch for site in eachindex(ϕ)
        ϕ[1, site] = staggered_kernel(U, ψ, site, mass, T, -1)
    end

    return nothing
end

function LinearAlgebra.mul!(
    ϕ::StaggeredFermionfield{CPU,T},
    D::Hermitian{StaggeredDiracOperator{CPU,T,TG,TF}},
    ψ::StaggeredFermionfield{CPU,T},
) where {T,TG,TF}
    temp = D.parent.temp
    mul!(temp, Daggered(D.parent), ψ) # temp = D†ψ
    mul!(ϕ, D.parent, temp) # ϕ = DD†ψ
    return nothing
end

@inline function staggered_kernel(U, ψ, site, mass, T, sgn=1)
    NX, NY, NZ, NT = dims(U)
    ϕₙ = 2mass * ψ[1, site]
    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    siteμ⁺ = move(site, 1, 1, NX)
    siteμ⁻ = move(site, 1, -1, NX)
    η = sgn * staggered_η(Val(1), site)
    ϕₙ += η * (cmvmul(U[1, site], ψ[1, siteμ⁺]) - cmvmul_d(U[1, siteμ⁻], ψ[1, siteμ⁻]))

    siteμ⁺ = move(site, 2, 1, NY)
    siteμ⁻ = move(site, 2, -1, NY)
    η = sgn * staggered_η(Val(2), site)
    ϕₙ += η * (cmvmul(U[2, site], ψ[1, siteμ⁺]) - cmvmul_d(U[2, siteμ⁻], ψ[1, siteμ⁻]))

    siteμ⁺ = move(site, 3, 1, NZ)
    siteμ⁻ = move(site, 3, -1, NZ)
    η = sgn * staggered_η(Val(3), site)
    ϕₙ += η * (cmvmul(U[3, site], ψ[1, siteμ⁺]) - cmvmul_d(U[3, siteμ⁻], ψ[1, siteμ⁻]))

    siteμ⁺ = move(site, 4, 1, NT)
    siteμ⁻ = move(site, 4, -1, NT)
    η = sgn * staggered_η(Val(4), site)
    ϕₙ += η * (cmvmul(U[4, site], ψ[1, siteμ⁺]) - cmvmul_d(U[4, siteμ⁻], ψ[1, siteμ⁻]))
    return T(0.5) * ϕₙ
end

# Use Val to reduce the amount of if-statements in the kernel
@inline staggered_η(::Val{1}, site) = 1
@inline staggered_η(::Val{2}, site) = ifelse(iseven(site[1]), 1, -1)
@inline staggered_η(::Val{3}, site) = ifelse(iseven(site[1] + site[2]), 1, -1)
@inline staggered_η(::Val{4}, site) = ifelse(iseven(site[1] + site[2] + site[3]), 1, -1)
@inline staggered_η(::Val{1i32}, site) = 1i32
@inline staggered_η(::Val{2i32}, site) = ifelse(iseven(site[1i32]), 1i32, -1i32)
@inline staggered_η(::Val{3i32}, site) =
    ifelse(iseven(site[1i32] + site[2i32]), 1i32, -1i32)
@inline staggered_η(::Val{4i32}, site) =
    ifelse(iseven(site[1i32] + site[2i32] + site[3i32]), 1i32, -1i32)

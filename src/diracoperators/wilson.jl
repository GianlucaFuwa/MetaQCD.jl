mutable struct WilsonDiracOperator{B,T,TG,TF} <: AbstractDiracOperator
    U::TG
    temp_F::TF
    mass::Float64
    function WilsonDiracOperator(U::Gaugefield{B,T}, mass) where {B,T}
        temp_F = Fermionfield(dims(U)...; BACKEND=B, T=T)
        return new{B,T,typeof(U),typeof(temp_F)}(U, temp_F, mass)
    end
end

const WilsonFermionfield{B,T,A} = Fermionfield{B,T,A,4}

function LinearAlgebra.mul!(
    ϕ::WilsonFermionfield{CPU,T},
    D::WilsonDiracOperator{CPU,T},
    ψ::WilsonFermionfield{CPU,T},
) where {T}
    U = D.U
    mass = T(D.mass)
    @assert dims(ϕ) == dims(ψ) == dims(U)

    @batch for site in eachindex(ϕ)
        for μ in 1:4
            ϕ[μ, site] = wilson_kernel(U, ψ, μ, site, mass, T)
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
    mass = T(D.parent.mass)
    @assert dims(ϕ) == dims(ψ) == dims(U)

    @batch for site in eachindex(ϕ)
        for μ in 1:4
            ϕ[μ, site] = wilson_kernel(U, ψ, μ, site, mass, T)
        end
    end

    return nothing
end

function LinearAlgebra.mul!(
    ϕ::WilsonFermionfield{CPU,T},
    D::Hermitian{WilsonDiracOperator{CPU,T,TG,TF}},
    ψ::WilsonFermionfield{CPU,T},
) where {T,TG,TF}
    temp = D.parent.temp_F
    mul!(temp, Daggered(D.parent), ψ) # temp = D†ψ
    mul!(ϕ, D.parent, temp) # ϕ = DD†ψ
    return nothing
end

@inline function wilson_kernel(U, ψ, site, mass, T, sgn=1)
    NX, NY, NZ, NT = dims(U)
    ϕₙ = (2mass + 8) * ψ[1, site]

    siteμ⁺ = move(site, 1, 1, NX)
    siteμ⁻ = move(site, 1, -1, NX)
    ϕₙ += (cmvmul(U[1, site], ψ[1, siteμ⁺]) - cmvmul_d(U[1, siteμ⁻], ψ[1, siteμ⁻]))

    siteμ⁺ = move(site, 2, 1, NY)
    siteμ⁻ = move(site, 2, -1, NY)
    ϕₙ += (cmvmul(U[2, site], ψ[1, siteμ⁺]) - cmvmul_d(U[2, siteμ⁻], ψ[1, siteμ⁻]))

    siteμ⁺ = move(site, 3, 1, NZ)
    siteμ⁻ = move(site, 3, -1, NZ)
    ϕₙ += (cmvmul(U[3, site], ψ[1, siteμ⁺]) - cmvmul_d(U[3, siteμ⁻], ψ[1, siteμ⁻]))

    siteμ⁺ = move(site, 4, 1, NT)
    siteμ⁻ = move(site, 4, -1, NT)
    ϕₙ += (cmvmul(U[4, site], ψ[1, siteμ⁺]) - cmvmul_d(U[4, siteμ⁻], ψ[1, siteμ⁻]))
    return T(0.5) * ϕₙ
end

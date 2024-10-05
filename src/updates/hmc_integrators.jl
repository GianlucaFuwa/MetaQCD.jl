Base.show(io::IO, ::MIME"text/plain", int::AbstractIntegrator) = print(io, "$(typeof(int))")
Base.show(io::IO, int::AbstractIntegrator) = print(io, "$(typeof(int))")

struct Leapfrog <: AbstractIntegrator end

function evolve!(::Leapfrog, U, hmc::HMC, fermion_action, bias)
    updateP!(U, hmc, 0.5, fermion_action, bias)

    for _ in 1:hmc.steps-1
        updateU!(U, hmc, 1.0)
        updateP!(U, hmc, 1.0, fermion_action, bias)
    end

    updateU!(U, hmc, 1.0)
    updateP!(U, hmc, 0.5, fermion_action, bias)
    return nothing
end

struct LeapfrogRA <: AbstractIntegrator 
    friction::Float64
    LeapfrogRA(friction) = new(friction)
end

function Base.show(io::IO, ::MIME"text/plain", int::LeapfrogRA)
    print(io, "$(typeof(int))(friction=$(int.friction))")
end

Base.show(io::IO, int::LeapfrogRA) = print(io, "$(typeof(int))(friction=$(int.friction))")

function evolve!(L::LeapfrogRA, U, hmc::HMC, fermion_action, bias)
    Δτ = hmc.Δτ

    for _ in 1:div(hmc.steps, 2)
        mul!(hmc.P, exp(0.5Δτ * L.friction))
        updateP!(U, hmc, 0.5, fermion_action, bias)
        updateU!(U, hmc, 1.0)
        updateP!(U, hmc, 0.5, fermion_action, bias)
        mul!(hmc.P, exp(0.5Δτ * L.friction))
    end

    for _ in 1:div(hmc.steps, 2)
        mul!(hmc.P, exp(-0.5Δτ * L.friction))
        updateP!(U, hmc, 0.5, fermion_action, bias)
        updateU!(U, hmc, 1.0)
        updateP!(U, hmc, 0.5, fermion_action, bias)
        mul!(hmc.P, exp(-0.5Δτ * L.friction))
    end

    return nothing
end

struct OMF2Slow <: AbstractIntegrator
    α::Float64
    β::Float64
    γ::Float64
    function OMF2Slow()
        α = 0.1931833275037836
        β = 0.5
        γ = 1.0 - 2.0 * α
        return new(α, β, γ)
    end
end

function evolve!(O2S::OMF2Slow, U, hmc::HMC, fermion_action, bias)
    for _ in 1:hmc.steps
        updateP!(U, hmc, O2S.α, fermion_action, bias)
        updateU!(U, hmc, O2S.β)
        updateP!(U, hmc, O2S.γ, fermion_action, bias)
        updateU!(U, hmc, O2S.β)
        updateP!(U, hmc, O2S.α, fermion_action, bias)
    end

    return nothing
end

struct OMF2 <: AbstractIntegrator
    α::Float64
    β::Float64
    γ::Float64
    function OMF2()
        α = 0.1931833275037836
        β = 0.5
        γ = 1.0 - 2.0 * α
        return new(α, β, γ)
    end
end

function evolve!(O2::OMF2, U, hmc::HMC, fermion_action, bias)
    updateP!(U, hmc, O2.α, fermion_action, bias)
    updateU!(U, hmc, O2.β)
    updateP!(U, hmc, O2.γ, fermion_action, bias)
    updateU!(U, hmc, O2.β)

    for _ in 1:hmc.steps-1
        updateP!(U, hmc, 2 * O2.α, fermion_action, bias)
        updateU!(U, hmc, O2.β)
        updateP!(U, hmc, O2.γ, fermion_action, bias)
        updateU!(U, hmc, O2.β)
    end

    updateP!(U, hmc, O2.α, fermion_action, bias)
    return nothing
end

struct OMF4Slow <: AbstractIntegrator
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
    μ::Float64
    ν::Float64
    function OMF4Slow()
        α = 0.08398315262876693
        β = 0.2539785108410595
        γ = 0.6822365335719091
        δ = -0.03230286765269967
        μ = 0.5 - γ - α
        ν = 1.0 - 2δ - 2β
        return new(α, β, γ, δ, μ, ν)
    end
end

function evolve!(O4S::OMF4Slow, U, hmc::HMC, fermion_action, bias)
    for _ in 1:hmc.steps
        updateP!(U, hmc, O4S.α, fermion_action, bias)
        updateU!(U, hmc, O4S.β)
        updateP!(U, hmc, O4S.γ, fermion_action, bias)
        updateU!(U, hmc, O4S.δ)

        updateP!(U, hmc, O4S.μ, fermion_action, bias)
        updateU!(U, hmc, O4S.ν)
        updateP!(U, hmc, O4S.μ, fermion_action, bias)

        updateU!(U, hmc, O4S.δ)
        updateP!(U, hmc, O4S.γ, fermion_action, bias)
        updateU!(U, hmc, O4S.β)
        updateP!(U, hmc, O4S.α, fermion_action, bias)
    end

    return nothing
end

struct OMF4 <: AbstractIntegrator
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
    μ::Float64
    ν::Float64
    function OMF4()
        α = 0.08398315262876693
        β = 0.2539785108410595
        γ = 0.6822365335719091
        δ = -0.03230286765269967
        μ = 0.5 - γ - α
        ν = 1.0 - 2δ - 2β
        return new(α, β, γ, δ, μ, ν)
    end
end

function evolve!(O4::OMF4, U, hmc::HMC, fermion_action, bias)
    updateP!(U, hmc, O4.α, fermion_action, bias)
    updateU!(U, hmc, O4.β)
    updateP!(U, hmc, O4.γ, fermion_action, bias)
    updateU!(U, hmc, O4.δ)

    updateP!(U, hmc, O4.μ, fermion_action, bias)
    updateU!(U, hmc, O4.ν)
    updateP!(U, hmc, O4.μ, fermion_action, bias)

    updateU!(U, hmc, O4.δ)
    updateP!(U, hmc, O4.γ, fermion_action, bias)
    updateU!(U, hmc, O4.β)

    for _ in 1:hmc.steps-1
        updateP!(U, hmc, 2 * O4.α, fermion_action, bias)
        updateU!(U, hmc, O4.β)
        updateP!(U, hmc, O4.γ, fermion_action, bias)
        updateU!(U, hmc, O4.δ)

        updateP!(U, hmc, O4.μ, fermion_action, bias)
        updateU!(U, hmc, O4.ν)
        updateP!(U, hmc, O4.μ, fermion_action, bias)

        updateU!(U, hmc, O4.δ)
        updateP!(U, hmc, O4.γ, fermion_action, bias)
        updateU!(U, hmc, O4.β)
    end

    updateP!(U, hmc, O4.α, fermion_action, bias)
    return nothing
end

# Repell-Attract Integrator: https://arxiv.org/pdf/2403.04607
struct OMF4RA <: AbstractIntegrator
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
    μ::Float64
    ν::Float64
    friction::Float64
    function OMF4RA(friction)
        α = 0.08398315262876693
        β = 0.2539785108410595
        γ = 0.6822365335719091
        δ = -0.03230286765269967
        μ = 0.5 - γ - α
        ν = 1.0 - 2δ - 2β
        return new(α, β, γ, δ, μ, ν, friction)
    end
end

Base.:-(O4::OMF4RA) = OMF4RA(-O4.friction)

function evolve!(O4::OMF4RA, U, hmc::HMC, fermion_action, bias)
    Δτ = hmc.Δτ
    mul!(hmc.P, exp(Δτ * O4.friction))
    updateP!(U, hmc, O4.α, fermion_action, bias)
    # mul!(hmc.P, exp(O4.α * Δτ * O4.friction))
    updateU!(U, hmc, O4.β)
    # mul!(hmc.P, exp(O4.γ * Δτ * O4.friction))
    updateP!(U, hmc, O4.γ, fermion_action, bias)
    # mul!(hmc.P, exp(O4.γ * Δτ * O4.friction))
    updateU!(U, hmc, O4.δ)

    # mul!(hmc.P, exp(O4.μ * Δτ * O4.friction))
    updateP!(U, hmc, O4.μ, fermion_action, bias)
    updateU!(U, hmc, O4.ν)
    # mul!(hmc.P, exp(O4.μ * Δτ * O4.friction))
    updateP!(U, hmc, O4.μ, fermion_action, bias)

    updateU!(U, hmc, O4.δ)
    # mul!(hmc.P, exp(O4.γ * Δτ * O4.friction))
    updateP!(U, hmc, O4.γ, fermion_action, bias)
    updateU!(U, hmc, O4.β)

    for i in 1:hmc.steps-1 
        sgn = i <= fld(hmc.steps-1, 2) ? 1 : -1
        mul!(hmc.P, exp(sgn * 2 * O4.α * Δτ * O4.friction))
        updateP!(U, hmc, 2 * O4.α, fermion_action, bias)
        updateU!(U, hmc, O4.β)
        mul!(hmc.P, exp(sgn * O4.γ * Δτ * O4.friction))
        updateP!(U, hmc, O4.γ, fermion_action, bias)
        updateU!(U, hmc, O4.δ)

        mul!(hmc.P, exp(sgn * O4.μ * Δτ * O4.friction))
        updateP!(U, hmc, O4.μ, fermion_action, bias)
        updateU!(U, hmc, O4.ν)
        mul!(hmc.P, exp(sgn * O4.μ * Δτ * O4.friction))
        updateP!(U, hmc, O4.μ, fermion_action, bias)

        updateU!(U, hmc, O4.δ)
        mul!(hmc.P, exp(sgn * O4.γ * Δτ * O4.friction))
        updateP!(U, hmc, O4.γ, fermion_action, bias)
        updateU!(U, hmc, O4.β)
    end

    # mul!(hmc.P, exp(-O4.α * Δτ * O4.friction))
    updateP!(U, hmc, O4.α, fermion_action, bias)
    mul!(hmc.P, exp(-Δτ * O4.friction))
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", int::OMF4RA)
    print(io, "$(typeof(int))(friction=$(int.friction))")
end
Base.show(io::IO, int::OMF4RA) = print(io, "$(typeof(int))(friction=$(int.friction))")

function integrator_from_str(str::String, friction)
    if str == "leapfrog" || str == "Leapfrog"
        return Leapfrog()
    elseif str == "leapfrogra" || str == "LeapfrogRA"
        return LeapfrogRA(friction)
    elseif str == "omf2" || str == "OMF2"
        return OMF2()
    elseif str == "omf2slow" || str == "OMF2Slow"
        return OMF2Slow()
    elseif str == "omf4" || str == "OMF4"
        return OMF4()
    elseif str == "omf4slow" || str == "OMF4Slow"
        return OMF4Slow()
    elseif str == "omf4ra" || str == "OMF4RA"
        return OMF4RA(friction)
    else
        error("integrator \"$(str)\" not supported")
    end
end

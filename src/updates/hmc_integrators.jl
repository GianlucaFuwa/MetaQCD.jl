struct Leapfrog <: AbstractIntegrator end

function evolve!(::Leapfrog, U, hmc::HMC, Bias)
    updateP!(U, hmc, 0.5, Bias)

    for _ = 1:hmc.steps-1
        updateU!(U, hmc, 1.0)
        updateP!(U, hmc, 1.0, Bias)
    end

    updateU!(U, hmc, 1.0)
    updateP!(U, hmc, 0.5, Bias)
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

function evolve!(O2S::OMF2Slow, U, hmc::HMC, Bias)

    for _ in 1:hmc.steps
        updateP!(U, hmc, O2S.α, Bias)
        updateU!(U, hmc, O2S.β)
        updateP!(U, hmc, O2S.γ, Bias)
        updateU!(U, hmc, O2S.β)
        updateP!(U, hmc, O2S.α, Bias)
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

function evolve!(O2::OMF2, U, hmc::HMC, Bias)
    updateP!(U, hmc, O2.α, Bias)
    updateU!(U, hmc, O2.β)
    updateP!(U, hmc, O2.γ, Bias)
    updateU!(U, hmc, O2.β)

    for _ in 1:hmc.steps-1
        updateP!(U, hmc, 2*O2.α, Bias)
        updateU!(U, hmc, O2.β)
        updateP!(U, hmc, O2.γ, Bias)
        updateU!(U, hmc, O2.β)
    end

    updateP!(U, hmc, O2.α, Bias)
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

function evolve!(O4S::OMF4Slow, U, hmc::HMC, Bias)

    for _ in 1:hmc.steps
        updateP!(U, hmc, O4S.α, Bias)
        updateU!(U, hmc, O4S.β)
        updateP!(U, hmc, O4S.γ, Bias)
        updateU!(U, hmc, O4S.δ)

        updateP!(U, hmc, O4S.μ, Bias)
        updateU!(U, hmc, O4S.ν)
        updateP!(U, hmc, O4S.μ, Bias)

        updateU!(U, hmc, O4S.δ)
        updateP!(U, hmc, O4S.γ, Bias)
        updateU!(U, hmc, O4S.β)
        updateP!(U, hmc, O4S.α, Bias)
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

function evolve!(O4::OMF4, U, hmc::HMC, Bias)
    updateP!(U, hmc, O4.α, Bias)
    updateU!(U, hmc, O4.β)
    updateP!(U, hmc, O4.γ, Bias)
    updateU!(U, hmc, O4.δ)

    updateP!(U, hmc, O4.μ, Bias)
    updateU!(U, hmc, O4.ν)
    updateP!(U, hmc, O4.μ, Bias)

    updateU!(U, hmc, O4.δ)
    updateP!(U, hmc, O4.γ, Bias)
    updateU!(U, hmc, O4.β)

    for _ in 1:hmc.steps-1
        updateP!(U, hmc, 2 * O4.α, Bias)
        updateU!(U, hmc, O4.β)
        updateP!(U, hmc, O4.γ, Bias)
        updateU!(U, hmc, O4.δ)

        updateP!(U, hmc, O4.μ, Bias)
        updateU!(U, hmc, O4.ν)
        updateP!(U, hmc, O4.μ, Bias)

        updateU!(U, hmc, O4.δ)
        updateP!(U, hmc, O4.γ, Bias)
        updateU!(U, hmc, O4.β)
    end

    updateP!(U, hmc, O4.α, Bias)
    return nothing
end

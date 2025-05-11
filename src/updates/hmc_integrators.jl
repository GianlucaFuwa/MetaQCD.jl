Base.show(io::IO, ::MIME"text/plain", int::AbstractIntegrator) = print(io, "$(typeof(int))")
Base.show(io::IO, int::AbstractIntegrator) = print(io, "$(typeof(int))")

function evolve!(U, hmc::HMC, fermion_action, bias, therm=Val(false))
    integrator = if therm == Val(true)
        default_integrator(hmc.levels[hmc.current_level[]].integrator)
    else
        hmc.levels[hmc.current_level[]].integrator
    end

    evolve!(integrator, U, hmc, fermion_action, bias, therm)
    return nothing
end

struct Leapfrog <: AbstractIntegrator end

num_U_updates(::Leapfrog) = 1

function evolve!(::Leapfrog, U, hmc::HMC, fermion_action, bias, therm)
    updateP!(U, hmc, 0.5, fermion_action, bias)

    for _ in 1:hmc.levels[hmc.current_level[]].numsteps-1
        updateU!(U, hmc, 1.0, fermion_action, bias, therm)
        updateP!(U, hmc, 1.0, fermion_action, bias)
    end

    updateU!(U, hmc, 1.0, fermion_action, bias, therm)
    updateP!(U, hmc, 0.5, fermion_action, bias)
    return nothing
end

struct LeapfrogRA <: AbstractIntegrator 
    friction::Float64
    LeapfrogRA(friction) = new(friction)
end

num_U_updates(::LeapfrogRA) = 1

Base.show(io::IO, ::MIME"text/plain", int::LeapfrogRA) =
    print(io, "$(typeof(int))(friction=$(int.friction))")
Base.show(io::IO, int::LeapfrogRA) = print(io, "$(typeof(int))(friction=$(int.friction))")

function evolve!(L::LeapfrogRA, U, hmc::HMC, fermion_action, bias, therm)
    Δτ = hmc.levels[hmc.current_level[]].Δτ

    # Repell
    for _ in 1:div(hmc.levels[hmc.current_level[]].numsteps, 2)
        mul!(hmc.P, exp(0.5Δτ * L.friction))
        updateP!(U, hmc, 0.5, fermion_action, bias)
        updateU!(U, hmc, 1.0, fermion_action, bias, therm)
        updateP!(U, hmc, 0.5, fermion_action, bias)
        mul!(hmc.P, exp(0.5Δτ * L.friction))
    end

    # Attract
    for _ in 1:div(hmc.levels[hmc.current_level[]].numsteps, 2)
        mul!(hmc.P, exp(-0.5Δτ * L.friction))
        updateP!(U, hmc, 0.5, fermion_action, bias)
        updateU!(U, hmc, 1.0, fermion_action, bias, therm)
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

num_U_updates(::OMF2Slow) = 2

function evolve!(O2S::OMF2Slow, U, hmc::HMC, fermion_action, bias, therm)
    for _ in 1:hmc.levels[hmc.current_level[]].numsteps
        updateP!(U, hmc, O2S.α, fermion_action, bias)
        updateU!(U, hmc, O2S.β, fermion_action, bias, therm)
        updateP!(U, hmc, O2S.γ, fermion_action, bias)
        updateU!(U, hmc, O2S.β, fermion_action, bias, therm)
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

num_U_updates(::OMF2) = 2

function evolve!(O2::OMF2, U, hmc::HMC, fermion_action, bias, therm)
    updateP!(U, hmc, O2.α, fermion_action, bias)
    updateU!(U, hmc, O2.β, fermion_action, bias, therm)
    updateP!(U, hmc, O2.γ, fermion_action, bias)
    updateU!(U, hmc, O2.β, fermion_action, bias, therm)

    for _ in 1:hmc.levels[hmc.current_level[]].numsteps-1
        updateP!(U, hmc, 2 * O2.α, fermion_action, bias)
        updateU!(U, hmc, O2.β, fermion_action, bias, therm)
        updateP!(U, hmc, O2.γ, fermion_action, bias)
        updateU!(U, hmc, O2.β, fermion_action, bias, therm)
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

num_U_updates(::OMF4Slow) = 5

function evolve!(O4S::OMF4Slow, U, hmc::HMC, fermion_action, bias, therm)
    for _ in 1:hmc.levels[hmc.current_level[]].numsteps
        updateP!(U, hmc, O4S.α, fermion_action, bias)
        updateU!(U, hmc, O4S.β, fermion_action, bias, therm)
        updateP!(U, hmc, O4S.γ, fermion_action, bias)
        updateU!(U, hmc, O4S.δ, fermion_action, bias, therm)

        updateP!(U, hmc, O4S.μ, fermion_action, bias)
        updateU!(U, hmc, O4S.ν, fermion_action, bias, therm)
        updateP!(U, hmc, O4S.μ, fermion_action, bias)

        updateU!(U, hmc, O4S.δ, fermion_action, bias, therm)
        updateP!(U, hmc, O4S.γ, fermion_action, bias)
        updateU!(U, hmc, O4S.β, fermion_action, bias, therm)
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

num_U_updates(::OMF4) = 5

function evolve!(O4::OMF4, U, hmc::HMC, fermion_action, bias, therm)
    updateP!(U, hmc, O4.α, fermion_action, bias)
    updateU!(U, hmc, O4.β, fermion_action, bias, therm)
    updateP!(U, hmc, O4.γ, fermion_action, bias)
    updateU!(U, hmc, O4.δ, fermion_action, bias, therm)

    updateP!(U, hmc, O4.μ, fermion_action, bias)
    updateU!(U, hmc, O4.ν, fermion_action, bias, therm)
    updateP!(U, hmc, O4.μ, fermion_action, bias)

    updateU!(U, hmc, O4.δ, fermion_action, bias, therm)
    updateP!(U, hmc, O4.γ, fermion_action, bias)
    updateU!(U, hmc, O4.β, fermion_action, bias, therm)

    for _ in 1:hmc.levels[hmc.current_level[]].numsteps-1
        updateP!(U, hmc, 2 * O4.α, fermion_action, bias)
        updateU!(U, hmc, O4.β, fermion_action, bias, therm)
        updateP!(U, hmc, O4.γ, fermion_action, bias)
        updateU!(U, hmc, O4.δ, fermion_action, bias, therm)

        updateP!(U, hmc, O4.μ, fermion_action, bias)
        updateU!(U, hmc, O4.ν, fermion_action, bias, therm)
        updateP!(U, hmc, O4.μ, fermion_action, bias)

        updateU!(U, hmc, O4.δ, fermion_action, bias, therm)
        updateP!(U, hmc, O4.γ, fermion_action, bias)
        updateU!(U, hmc, O4.β, fermion_action, bias, therm)
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

num_U_updates(::OMF4RA) = 5

function evolve!(O4::OMF4RA, U, hmc::HMC, fermion_action, bias, therm)
    Δτ = hmc.levels[hmc.current_level[]].Δτ

    for _ in 1:div(hmc.levels[hmc.current_level[]].numsteps, 2)
        mul!(hmc.P, exp(Δτ * O4.friction))
        updateP!(U, hmc, O4.α, fermion_action, bias)
        updateU!(U, hmc, O4.β, fermion_action, bias, therm)
        updateP!(U, hmc, O4.γ, fermion_action, bias)
        updateU!(U, hmc, O4.δ, fermion_action, bias, therm)

        updateP!(U, hmc, O4.μ, fermion_action, bias)
        updateU!(U, hmc, O4.ν, fermion_action, bias, therm)
        updateP!(U, hmc, O4.μ, fermion_action, bias)

        updateU!(U, hmc, O4.δ, fermion_action, bias, therm)
        updateP!(U, hmc, O4.γ, fermion_action, bias)
        updateU!(U, hmc, O4.β, fermion_action, bias, therm)
        updateP!(U, hmc, O4.α, fermion_action, bias)
        mul!(hmc.P, exp(Δτ * O4.friction))
    end

    for _ in 1:div(hmc.levels[hmc.current_level[]].numsteps, 2)
        mul!(hmc.P, exp(-Δτ * O4.friction))
        updateP!(U, hmc, O4.α, fermion_action, bias)
        updateU!(U, hmc, O4.β, fermion_action, bias, therm)
        updateP!(U, hmc, O4.γ, fermion_action, bias)
        updateU!(U, hmc, O4.δ, fermion_action, bias, therm)

        updateP!(U, hmc, O4.μ, fermion_action, bias)
        updateU!(U, hmc, O4.ν, fermion_action, bias, therm)
        updateP!(U, hmc, O4.μ, fermion_action, bias)

        updateU!(U, hmc, O4.δ, fermion_action, bias, therm)
        updateP!(U, hmc, O4.γ, fermion_action, bias)
        updateU!(U, hmc, O4.β, fermion_action, bias, therm)
        updateP!(U, hmc, O4.α, fermion_action, bias)
        mul!(hmc.P, exp(-Δτ * O4.friction))
    end

    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", int::OMF4RA)
    print(io, "$(typeof(int))(friction=$(int.friction))")
end
Base.show(io::IO, int::OMF4RA) = print(io, "$(typeof(int))(friction=$(int.friction))")

function integrator_from_str(str::String, friction=0.0)
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

@inline default_integrator(int::AbstractIntegrator) = int
@inline default_integrator(int::LeapfrogRA) = Leapfrog()
@inline default_integrator(int::OMF4RA) = OMF4()

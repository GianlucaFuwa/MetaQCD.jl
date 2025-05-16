Base.show(io::IO, ::MIME"text/plain", int::AbstractIntegrator) = print(io, "$(typeof(int))")
Base.show(io::IO, int::AbstractIntegrator) = print(io, "$(typeof(int))")

function evolve!(U, hmc::HMC, fermion_action, bias, therm=Val(false), level=1)
    @assert level >= 1 && level <= _unwrap_val(hmc.numlevels) """
    Invalid level: $level (must be between 1 and $(_unwrap_val(hmc.numlevels)))
    """

    integrator = if therm == Val(true)
        default_integrator(hmc.levels[level].integrator)
    else
        hmc.levels[level].integrator
    end

    evolve!(integrator, U, hmc, fermion_action, bias, therm, level)
    return nothing
end

struct Leapfrog <: AbstractIntegrator end

function evolve!(::Leapfrog, U, hmc::HMC, fermion_action, bias, therm, level)
    updateP!(U, hmc, 0.5, fermion_action, bias, level)

    for _ in 1:hmc.levels[level].numsteps-1
        updateU!(U, hmc, 1.0, fermion_action, bias, therm, level)
        updateP!(U, hmc, 1.0, fermion_action, bias, level)
    end

    updateU!(U, hmc, 1.0, fermion_action, bias, therm, level)
    updateP!(U, hmc, 0.5, fermion_action, bias, level)
    return nothing
end

struct LeapfrogRA <: AbstractIntegrator
    friction::Float64
    LeapfrogRA(friction) = new(friction)
end

function Base.show(io::IO, ::MIME"text/plain", int::LeapfrogRA)
    return print(io, "$(typeof(int))(friction=$(int.friction))")
end

Base.show(io::IO, int::LeapfrogRA) = print(io, "$(typeof(int))(friction=$(int.friction))")

function evolve!(L::LeapfrogRA, U, hmc::HMC, fermion_action, bias, therm, level)
    Δτ = hmc.levels[level].Δτ

    # Repell
    for _ in 1:div(hmc.levels[level].numsteps, 2)
        mul!(hmc.P, exp(0.5Δτ * L.friction))
        updateP!(U, hmc, 0.5, fermion_action, bias, level)
        updateU!(U, hmc, 1.0, fermion_action, bias, therm, level)
        updateP!(U, hmc, 0.5, fermion_action, bias, level)
        mul!(hmc.P, exp(0.5Δτ * L.friction))
    end

    # Attract
    for _ in 1:div(hmc.levels[level].numsteps, 2)
        mul!(hmc.P, exp(-0.5Δτ * L.friction))
        updateP!(U, hmc, 0.5, fermion_action, bias, level)
        updateU!(U, hmc, 1.0, fermion_action, bias, therm, level)
        updateP!(U, hmc, 0.5, fermion_action, bias, level)
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

function evolve!(O2S::OMF2Slow, U, hmc::HMC, fermion_action, bias, therm, level)
    for _ in 1:hmc.levels[level].numsteps
        updateP!(U, hmc, O2S.α, fermion_action, bias, level)
        updateU!(U, hmc, O2S.β, fermion_action, bias, therm, level)
        updateP!(U, hmc, O2S.γ, fermion_action, bias, level)
        updateU!(U, hmc, O2S.β, fermion_action, bias, therm, level)
        updateP!(U, hmc, O2S.α, fermion_action, bias, level)
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

function evolve!(O2::OMF2, U, hmc::HMC, fermion_action, bias, therm, level)
    updateP!(U, hmc, O2.α, fermion_action, bias, level)
    updateU!(U, hmc, O2.β, fermion_action, bias, therm, level)
    updateP!(U, hmc, O2.γ, fermion_action, bias, level)
    updateU!(U, hmc, O2.β, fermion_action, bias, therm, level)

    for _ in 1:hmc.levels[level].numsteps-1
        updateP!(U, hmc, 2 * O2.α, fermion_action, bias, level)
        updateU!(U, hmc, O2.β, fermion_action, bias, therm, level)
        updateP!(U, hmc, O2.γ, fermion_action, bias, level)
        updateU!(U, hmc, O2.β, fermion_action, bias, therm, level)
    end

    updateP!(U, hmc, O2.α, fermion_action, bias, level)
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

function evolve!(O4S::OMF4Slow, U, hmc::HMC, fermion_action, bias, therm, level)
    for _ in 1:hmc.levels[level].numsteps
        updateP!(U, hmc, O4S.α, fermion_action, bias, level)
        updateU!(U, hmc, O4S.β, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4S.γ, fermion_action, bias, level)
        updateU!(U, hmc, O4S.δ, fermion_action, bias, therm, level)

        updateP!(U, hmc, O4S.μ, fermion_action, bias, level)
        updateU!(U, hmc, O4S.ν, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4S.μ, fermion_action, bias, level)

        updateU!(U, hmc, O4S.δ, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4S.γ, fermion_action, bias, level)
        updateU!(U, hmc, O4S.β, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4S.α, fermion_action, bias, level)
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

function evolve!(O4::OMF4, U, hmc::HMC, fermion_action, bias, therm, level)
    updateP!(U, hmc, O4.α, fermion_action, bias, level)
    updateU!(U, hmc, O4.β, fermion_action, bias, therm, level)
    updateP!(U, hmc, O4.γ, fermion_action, bias, level)
    updateU!(U, hmc, O4.δ, fermion_action, bias, therm, level)

    updateP!(U, hmc, O4.μ, fermion_action, bias, level)
    updateU!(U, hmc, O4.ν, fermion_action, bias, therm, level)
    updateP!(U, hmc, O4.μ, fermion_action, bias, level)

    updateU!(U, hmc, O4.δ, fermion_action, bias, therm, level)
    updateP!(U, hmc, O4.γ, fermion_action, bias, level)
    updateU!(U, hmc, O4.β, fermion_action, bias, therm, level)

    for _ in 1:hmc.levels[level].numsteps-1
        updateP!(U, hmc, 2 * O4.α, fermion_action, bias, level)
        updateU!(U, hmc, O4.β, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4.γ, fermion_action, bias, level)
        updateU!(U, hmc, O4.δ, fermion_action, bias, therm, level)

        updateP!(U, hmc, O4.μ, fermion_action, bias, level)
        updateU!(U, hmc, O4.ν, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4.μ, fermion_action, bias, level)

        updateU!(U, hmc, O4.δ, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4.γ, fermion_action, bias, level)
        updateU!(U, hmc, O4.β, fermion_action, bias, therm, level)
    end

    updateP!(U, hmc, O4.α, fermion_action, bias, level)
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

function evolve!(O4::OMF4RA, U, hmc::HMC, fermion_action, bias, therm, level)
    Δτ = hmc.levels[level].Δτ

    for _ in 1:div(hmc.levels[level].numsteps, 2)
        mul!(hmc.P, exp(Δτ * O4.friction))
        updateP!(U, hmc, O4.α, fermion_action, bias, level)
        updateU!(U, hmc, O4.β, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4.γ, fermion_action, bias, level)
        updateU!(U, hmc, O4.δ, fermion_action, bias, therm, level)

        updateP!(U, hmc, O4.μ, fermion_action, bias, level)
        updateU!(U, hmc, O4.ν, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4.μ, fermion_action, bias, level)

        updateU!(U, hmc, O4.δ, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4.γ, fermion_action, bias, level)
        updateU!(U, hmc, O4.β, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4.α, fermion_action, bias, level)
        mul!(hmc.P, exp(Δτ * O4.friction))
    end

    for _ in 1:div(hmc.levels[level].numsteps, 2)
        mul!(hmc.P, exp(-Δτ * O4.friction))
        updateP!(U, hmc, O4.α, fermion_action, bias, level)
        updateU!(U, hmc, O4.β, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4.γ, fermion_action, bias, level)
        updateU!(U, hmc, O4.δ, fermion_action, bias, therm, level)

        updateP!(U, hmc, O4.μ, fermion_action, bias, level)
        updateU!(U, hmc, O4.ν, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4.μ, fermion_action, bias, level)

        updateU!(U, hmc, O4.δ, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4.γ, fermion_action, bias, level)
        updateU!(U, hmc, O4.β, fermion_action, bias, therm, level)
        updateP!(U, hmc, O4.α, fermion_action, bias, level)
        mul!(hmc.P, exp(-Δτ * O4.friction))
    end

    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", int::OMF4RA)
    return print(io, "$(typeof(int))(friction=$(int.friction))")
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

function num_U_updates(str::String)
    if lower_case(str) ∈ ("leapfrog", "leapfrogra")
        return 1
    elseif lower_case(str) ∈ ("omf2", "omf2slow")
        return 2
    elseif lower_case(str) ∈ ("omf4", "omf4slow", "omf4ra")
        return 5
    else
        error("integrator \"$(str)\" not supported")
    end
end

@inline default_integrator(int::AbstractIntegrator) = int
@inline default_integrator(int::LeapfrogRA) = Leapfrog()
@inline default_integrator(int::OMF4RA) = OMF4()

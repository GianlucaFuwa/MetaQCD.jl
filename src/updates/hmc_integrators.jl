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

    return evolve!(integrator, U, hmc, fermion_action, bias, therm, level)
end

struct Leapfrog <: AbstractIntegrator end

function evolve!(::Leapfrog, U, hmc::HMC, fermion_action, bias, therm, level)
    updateP!(U, hmc, 0.5, fermion_action, bias, level)

    for _ in 1:hmc.levels[level].numsteps-1
        updateU!(U, hmc, 1.0, fermion_action, bias, therm, level)
        updateP!(U, hmc, 1.0, fermion_action, bias, level, true)
    end

    updateU!(U, hmc, 1.0, fermion_action, bias, therm, level)
    updateP!(U, hmc, 0.5, fermion_action, bias, level, true)
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
        updateP!(U, hmc, 0.5, fermion_action, bias, level, true)
        mul!(hmc.P, exp(0.5Δτ * L.friction))
    end

    # Attract
    for _ in 1:div(hmc.levels[level].numsteps, 2)
        mul!(hmc.P, exp(-0.5Δτ * L.friction))
        updateP!(U, hmc, 0.5, fermion_action, bias, level)
        updateU!(U, hmc, 1.0, fermion_action, bias, therm, level)
        updateP!(U, hmc, 0.5, fermion_action, bias, level, true)
        mul!(hmc.P, exp(-0.5Δτ * L.friction))
    end

    return nothing
end

mutable struct LeapfrogConstrained <: AbstractIntegrator
    interval::NTuple{2,Float64}
    velocity::Float64
    LeapfrogConstrained(interval, velocity) = new(interval, velocity)
end

function Base.show(io::IO, ::MIME"text/plain", int::LeapfrogConstrained)
    return print(io, "$(typeof(int))(interval=$(int.interval), velocity=$(int.velocity))")
end

Base.show(io::IO, int::LeapfrogConstrained) = print(io, "$(typeof(int))(interval=$(int.interval))")

function evolve!(L::LeapfrogConstrained, U, hmc::HMC, fermion_action, bias, forward, level)
    @assert length(bias) == 1 "Constrained evolution only supported for a single CV"
    Z, Z̃ = L.interval
    Δτ = hmc.levels[level].Δτ
    T = Δτ * hmc.levels[level].numsteps
    ci = Z
    ci_old = Z
    c(t) = Z + (Z̃ - Z)*0.5(1 - cos(π*t/T))
    # c(t) = Z + (Z̃ - Z)*t/2
    λ = 0.0
    t = 0.0
    cv_vec = Float64[]
    W_vec = Float64[]

    for _ in 1:hmc.levels[level].numsteps
        t += Δτ
        ci = c(t)
        updateP!(U, hmc, 0.5, fermion_action, bias, level)
        copy!(hmc.P0, hmc.P) # P0 -> P_0 - Δτ/2 * ∇S

        # SHAKE -- solve for λ
        λ = solve_via_secant!(U, hmc, bias, ci; Δτ, λ) # force -> C(U1)
        W = -λ * (ci - ci_old)
        push!(cv_vec, ci)
        push!(W_vec, W)
        # @show W
        # @show λ

        fac = 1/real(-6dot(hmc.force2, hmc.force2))
        dc_dot_p = real(-6dot(hmc.force2, hmc.P))

        updateP!(U, hmc, 0.5, fermion_action, bias, level) # force -> dS/dU1
        dc_dot_ds = real(-6dot(hmc.force2, hmc.force))

        # RATTLE -- solve for μ
        μ = -fac * (dc_dot_p + Δτ/2 * dc_dot_ds)
        # @show μ
        add!(hmc.P, hmc.force2, μ)

        cons = real(-6dot(hmc.force2, hmc.P))
        @assert abs(cons) < 1e-10 "hidden constraint was $cons"

        ci_old = ci
    end

    return cv_vec, W_vec
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
        updateP!(U, hmc, O2S.α, fermion_action, bias, level, true)
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
        updateP!(U, hmc, 2 * O2.α, fermion_action, bias, level, true)
        updateU!(U, hmc, O2.β, fermion_action, bias, therm, level)
        updateP!(U, hmc, O2.γ, fermion_action, bias, level)
        updateU!(U, hmc, O2.β, fermion_action, bias, therm, level)
    end

    updateP!(U, hmc, O2.α, fermion_action, bias, level, true)
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
        updateP!(U, hmc, O4S.α, fermion_action, bias, level, true)
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
        updateP!(U, hmc, 2 * O4.α, fermion_action, bias, level, true)
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

    updateP!(U, hmc, O4.α, fermion_action, bias, level, true)
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
        updateP!(U, hmc, O4.α, fermion_action, bias, level, true)
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
        updateP!(U, hmc, O4.α, fermion_action, bias, level, true)
        mul!(hmc.P, exp(-Δτ * O4.friction))
    end

    return nothing
end

mutable struct OMF4Constrained <: AbstractIntegrator
    interval::NTuple{2,Float64}
    velocity::Float64
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
    μ::Float64
    ν::Float64
    function OMF4Constrained(interval, velocity)
        α = 0.08398315262876693
        β = 0.2539785108410595
        γ = 0.6822365335719091
        δ = -0.03230286765269967
        μ = 0.5 - γ - α
        ν = 1.0 - 2δ - 2β
        return new(interval, velocity, α, β, γ, δ, μ, ν)
    end
end

include("sched.jl")

function evolve!(O4C::OMF4Constrained, U, hmc::HMC, fermion_action, bias, therm, level)
    @assert level == 1
    @assert length(bias) == 1 "Constrained evolution only supported for a single CV"
    Z, Z̃ = O4C.interval
    Δτ = hmc.levels[level].Δτ
    v = O4C.velocity
    numsteps = maximum((hmc.levels[level].numsteps, cld(abs(Z̃ - Z), v)))
    # @show Z, Z̃
    # @show numsteps
    T = Δτ * numsteps
    # @show T
    # c(t) = optimal_schedule(bias, Z, Z̃).z(t/T)
    c(t) = Z + (Z̃ - Z)*0.5(1 - cos(π*t/T))
    λ = 0.0
    t = 0.0
    W = 0.0
    ci_old = c(0.0)

    for i in 1:numsteps
        t += O4C.β*Δτ
        ci = c(t)
        # @show ci - ci_old
        updateP!(U, hmc, O4C.α, fermion_action, bias, level)
        copy!(hmc.P0, hmc.P)
        λ = solve_via_secant!(U, hmc, bias, ci; Δτ=O4C.β*Δτ, λ)
        W += λ * (ci - ci_old)
        ci_old = ci
        updateP!(U, hmc, O4C.γ, fermion_action, bias, level)
        copy!(hmc.P0, hmc.P)
        t += O4C.δ*Δτ
        ci = c(t)
        # @show ci - ci_old
        λ = solve_via_secant!(U, hmc, bias, ci; Δτ=O4C.δ*Δτ, λ)
        W += λ * (ci - ci_old)
        ci_old = ci

        updateP!(U, hmc, O4C.μ, fermion_action, bias, level)
        copy!(hmc.P0, hmc.P)
        t += O4C.ν*Δτ
        ci = c(t)
        # @show ci - ci_old
        λ = solve_via_secant!(U, hmc, bias, ci; Δτ=O4C.ν*Δτ, λ)
        W += λ * (ci - ci_old)
        ci_old = ci
        updateP!(U, hmc, O4C.μ, fermion_action, bias, level)
        copy!(hmc.P0, hmc.P)

        t += O4C.δ*Δτ
        ci = c(t)
        # @show ci - ci_old
        λ = solve_via_secant!(U, hmc, bias, ci; Δτ=O4C.δ*Δτ, λ)
        W += λ * (ci - ci_old)
        ci_old = ci
        updateP!(U, hmc, O4C.γ, fermion_action, bias, level)
        copy!(hmc.P0, hmc.P)
        t += O4C.β*Δτ
        ci = c(t)
        # @show ci - ci_old
        λ = solve_via_secant!(U, hmc, bias, ci; Δτ=O4C.β*Δτ, λ)
        W += λ * (ci - ci_old)
        ci_old = ci
        # @show t

        fac = 1/real(-6dot(hmc.force2, hmc.force2))
        dc_dot_p = real(-6dot(hmc.force2, hmc.P))
        updateP!(U, hmc, O4C.α, fermion_action, bias, level, true)
        dc_dot_ds = real(-6dot(hmc.force2, hmc.force))
        μ = -fac * (dc_dot_p + Δτ*O4C.α * dc_dot_ds)
        add!(hmc.P, hmc.force2, μ)
    end

    return -W
end

function Base.show(io::IO, ::MIME"text/plain", int::OMF4RA)
    return print(io, "$(typeof(int))(friction=$(int.friction), velocity=$(int.velocity))")
end
Base.show(io::IO, int::OMF4RA) = print(io, "$(typeof(int))(friction=$(int.friction))")

function integrator_from_str(str::String, friction=0.0, velocity=0.1, constraint=nothing)
    if lowercase(str) == "Leapfrog"
        return Leapfrog()
    elseif lowercase(str) == "leapfrogra"
        return LeapfrogRA(friction)
    elseif lowercase(str) == "leapfrogconstrained"
        itvl = isnothing(constraint) ? (0.0, 1.0) : (constraint, constraint)
        return LeapfrogConstrained(itvl, velocity)
    elseif lowercase(str) == "omf2"
        return OMF2()
    elseif lowercase(str) == "omf2slow"
        return OMF2Slow()
    elseif lowercase(str) == "omf4"
        return OMF4()
    elseif lowercase(str) == "omf4slow"
        return OMF4Slow()
    elseif lowercase(str) == "omf4ra" || str == "OMF4RA"
        return OMF4RA(friction)
    elseif lowercase(str) == "omf4constrained"
        itvl = isnothing(constraint) ? (0.0, 1.0) : (constraint, constraint)
        return OMF4Constrained(itvl, velocity)
    else
        error("integrator \"$(str)\" not supported")
    end
end

function num_U_updates(str::String)
    if lowercase(str) ∈ ("leapfrog", "leapfrogra", "leapfrogconstrained")
        return 1
    elseif lowercase(str) ∈ ("omf2", "omf2slow")
        return 2
    elseif lowercase(str) ∈ ("omf4", "omf4slow", "omf4ra", "omf4constrained")
        return 5
    else
        error("integrator \"$(str)\" not supported")
    end
end

function num_P_updates(str::String, numsteps)
    if lowercase(str) ∈ ("leapfrog", "leapfrogra", "leapfrogconstrained")
        return numsteps + 1
    elseif lowercase(str) == "omf2slow"
        return 3numsteps
    elseif lowercase(str) == "omf2"
        return 2(numsteps-1) + 3
    elseif lowercase(str) ∈ ("omf4slow", "omf4constrained")
        return 6numsteps
    elseif lowercase(str) ∈ ("omf4", "omf4ra")
        return 5(numsteps-1) + 6
    else
        error("integrator \"$(str)\" not supported")
    end
end

is_constrained(integrator) = integrator isa ConstrainedIntegrator

const ConstrainedIntegrator = Union{LeapfrogConstrained,OMF4Constrained}

@inline default_integrator(int::AbstractIntegrator) = int
@inline default_integrator(::LeapfrogRA) = Leapfrog()
@inline default_integrator(::LeapfrogConstrained) = Leapfrog()
@inline default_integrator(::OMF4RA) = OMF4()
@inline default_integrator(::OMF4Constrained) = OMF4()

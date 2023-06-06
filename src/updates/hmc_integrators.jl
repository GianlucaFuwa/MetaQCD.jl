struct Leapfrog <: AbstractIntegrator
end

function (lf::Leapfrog)(U::T1, method::T2, Bias) where {T1<:Gaugefield, T2<:HMCUpdate}
    updateP!(U, method, 0.5, Bias)

    for i = 1:method.steps-1
        updateU!(U, method, 1.0)
        updateP!(U, method, 1.0, Bias)
    end

    updateU!(U, method, 1.0)
    updateP!(U, method, 0.5, Bias)
    return nothing
end

struct OMF2Slow <: AbstractIntegrator
    α::Float64
    β::Float64
    γ::Float64

    function OMF2_slow()
        α = 0.1931833275037836
        β = 0.5
        γ = 1.0 - 2.0 * α
        return new(α, β, γ)
    end
end

function (O2S::OMF2Slow)(U::T1, method::T2, Bias) where {T1<:Gaugefield, T2<:HMCUpdate}

    for i in 1:method.steps
        updateP!(U, method, O2S.α, Bias)
        updateU!(U, method, O2S.β)
        updateP!(U, method, O2S.γ, Bias)
        updateU!(U, method, O2S.β)
        updateP!(U, method, O2S.α, Bias)
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

function (O2::OMF2)(U::T1, method::T2, Bias) where {T1<:Gaugefield, T2<:HMCUpdate}
    updateP!(U, method, O2.α, Bias)
    updateU!(U, method, O2.β)
    updateP!(U, method, O2.γ, Bias)
    updateU!(U, method, O2.β)

    for i in 1:method.steps-1
        updateP!(U, method, 2*O2.α, Bias)
        updateU!(U, method, O2.β)
        updateP!(U, method, O2.γ, Bias)
        updateU!(U, method, O2.β)
    end

    updateP!(U, method, O2.α, Bias)
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
        ν = 1.0 - 2.0 * δ - 2.0 * β
        return new(α, β, γ, δ, μ, ν)
    end
end

function (O4S::OMF4Slow)(U::T1, method::T2, Bias) where {T1<:Gaugefield, T2<:HMCUpdate}

    for i in 1:method.steps
        updateP!(U, method, O4S.α, Bias)
        updateU!(U, method, O4S.β)
        updateP!(U, method, O4S.γ, Bias)
        updateU!(U, method, O4S.δ)

        updateP!(U, method, O4S.μ, Bias)
        updateU!(U, method, O4S.ν)
        updateP!(U, method, O4S.μ, Bias)

        updateU!(U, method, O4S.δ)
        updateP!(U, method, O4S.γ, Bias)
        updateU!(U, method, O4S.β)
        updateP!(U, method, O4S.α, Bias)
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
        ν = 1.0 - 2.0 * δ - 2.0 * β
        return new(α, β, γ, δ, μ, ν)
    end
end

function (O4::OMF4)(U::T1, method::T2, Bias) where {T1<:Gaugefield, T2<:HMCUpdate}
    updateP!(U, method, O4.α, Bias)
    updateU!(U, method, O4.β)
    updateP!(U, method, O4.γ, Bias)
    updateU!(U, method, O4.δ)

    updateP!(U, method, O4.μ, Bias)
    updateU!(U, method, O4.ν)
    updateP!(U, method, O4.μ, Bias)

    updateU!(U, method, O4.δ)
    updateP!(U, method, O4.γ, Bias)
    updateU!(U, method, O4.β)

    for i in 1:method.steps-1
        updateP!(U, method, 2 * O4.α, Bias)
        updateU!(U, method, O4.β)
        updateP!(U, method, O4.γ, Bias)
        updateU!(U, method, O4.δ)

        updateP!(U, method, O4.μ, Bias)
        updateU!(U, method, O4.ν)
        updateP!(U, method, O4.μ, Bias)

        updateU!(U, method, O4.δ)
        updateP!(U, method, O4.γ, Bias)
        updateU!(U, method, O4.β)
    end

    updateP!(U, method, O4.α, Bias)
    return nothing
end
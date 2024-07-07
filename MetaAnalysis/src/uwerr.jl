struct UWerr{T<:Function} <: AbstractErrorEstimator
    wpm::Dict{String,Vector{Float64}}
    id::String
    func::T
    function UWerr(; fixed_W=-1.0, S_τ=-1.0, Γ_ratio=-1.0, τ_exp=-1.0, func=identity)
        @assert length(findall(x -> x>0, (fixed_W, S_τ, Γ_ratio, τ_exp))) <= 1
        id = "$(rand(Int64))"
        wpm = Dict{String,Vector{Float64}}()
        wpm[id] = [fixed_W, S_τ, Γ_ratio, τ_exp]
        return new{typeof(func)}(wpm, id, func)
    end
end

function (u::UWerr)(x, weights=nothing)
    @assert weights === nothing "UWerr only supported without weights"
    obs = uwreal(u.func.(x), u.id)
    uwerr(obs, u.wpm)
    return value(obs), ADerrors.err(obs), ADerrors.taui(obs, u.id)
end

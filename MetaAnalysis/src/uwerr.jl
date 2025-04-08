struct UWerr{T<:Function} <: AbstractErrorEstimator
    wpm::Dict{String,Vector{Float64}}
    id::String
    func::T
    function UWerr(
        ; fixed_w=-1.0, s_tau=-1.0, gamma_ratio=-1.0, tau_exp=-1.0, func=identity
    )
        @assert length(findall(x -> x>0, (fixed_w, s_tau, gamma_ratio, tau_exp))) <= 1
        id = "$(rand(UInt64))"
        wpm = Dict{String,Vector{Float64}}()
        wpm[id] = [fixed_w, s_tau, gamma_ratio, tau_exp]
        return new{typeof(func)}(wpm, id, func)
    end
end

function (u::UWerr)(x, weights=nothing)
    @assert isnothing(weights) "UWerr only supported without weights"
    obs = uwreal(u.func.(x), u.id)
    uwerr(obs, u.wpm)
    return value(obs), ADerrors.err(obs), ADerrors.taui(obs, u.id)
end

function clear_wspace!()
    global ADerrors.wsg = ADerrors.wspace(
        similar(Vector{ADerrors.fbd}, 0),
        0,
        similar(Vector{Int64}, 0),
        Dict{Int64, Int64}(),
        Dict{Int64, String}(), Dict{String, Int64}(),
        Dict{Int64, Vector{String}}(),
        Dict{Int64, Vector{Int64}}(),
        -12345,
    )
end

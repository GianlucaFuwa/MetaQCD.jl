struct Jackknife{T<:Function} <: AbstractErrorEstimator
    blocksize::Int64
    τ::Int64
    func::T
    function Jackknife(; blocksize=0, τ=0, func=mean)
        @assert blocksize >= 0
        @assert τ >= 0
        return new{typeof(func)}(blocksize, τ, func)
    end
end

function (j::Jackknife)(x::Vector{<:Real}, weights=nothing)
    N = length(x)

    if j.τ == 0
        τ = autoc_time_int(x)
        if j.blocksize == 0
            B = round(Int64, 2τ, RoundNearestTiesAway)
        else
            B = j.blocksize
        end
    else
        τ = j.τ
        B = Int64(2j.τ)
    end

    M = fld(N, B)
    A = zeros(Float64, M + 1)

    if weights === nothing
        A[1] = j.func(x)

        p = Progress(M, desc = "Jackknifing...")
        @batch threadlocal=zeros(Float64, N - B)::Vector{Float64} for i in 2:M+1
            indices = (B * (i-2) + 1):(B * (i-1))
            jack_subset!(threadlocal, x, indices)
            A[i] = j.func(threadlocal)
            next!(p)
        end

        meanA = A[1]
        stdA = sqrt((M - 1)^2 / M * var(A))
    else
        @assert length(weights) == N "length(weights) has to be equal to length(x)"
        @info "Weighted works for means, i.e., b.func = mean"
        x .*= weights
        A[1] = sum(x) / sum(weights)

        p = Progress(M, desc="Bootstrapping...")
        @batch threadlocal=zeros(Int64, N - B)::Vector{Int64} for i = 2:M+1
            tmpE = 0.0
            tmpD = 0.0
            indices = (B * (i-2) + 1):(B * (i-1))
            jack_subset!(threadlocal, x, indices)
            tmpE += sum(threadlocal)
            tmpD += sum(view(weights, indices))
            A[i] = tmpE / tmpD
            next!(p)
        end

        meanA = A[1]
        stdA = sqrt((M - 1)^2 / M * var(A))
    end

    return meanA, stdA, τ
end

function jack_subset!(dest, src, indices)
    @assert length(dest) == length(src)-length(indices)

    i = 1

    for (j, el) in enumerate(src)
        j ∈ indices && continue
        dest[i] = el
        i += 1
    end
    
    return nothing
end

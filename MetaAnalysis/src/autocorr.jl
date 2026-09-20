function autoc_time_int(x::Vector{<:Real})
    M = length(x)
    avg = mean(x)

    C = Γ₀ = zero(Float64) 

    for k in 1:(M - 1)
        Γ₀ += (x[k] - avg) * (x[k + 1] - avg)
    end

    Γ₀ /= M - 1

    for t in 2:M-1
        tmp = 0.0

        for k in 1:(M - t)
            tmp += (x[k] - avg) * (x[k + t] - avg)
        end
    
        tmp_n = tmp / (M - t)

        (tmp_n <= 0) && break

        C += tmp_n
    end

    C = Γ₀ + 2C
    τ = C / 2Γ₀
    return τ
end

function autoc_time_int_jackknife(x::Vector{<:Real}; block_size=nothing)
    M = length(x)
    # full-sample estimate
    τ_full = try
        round(Int, autoc_time_int(x))
    catch _
        1
    end
    block_size = isnothing(block_size) ? min(4τ_full, div(M, 2)) : block_size

    nblocks = M ÷ block_size
    @assert nblocks > 1 "need more than one block"

    Mtrim = nblocks * block_size
    xs = @view x[1:Mtrim]

    # leave-one-block-out estimates
    τ_jk = Vector{Float64}(undef, nblocks)

    for b in 1:nblocks
        idx = vcat(1:(b-1)*block_size, (b*block_size+1):Mtrim)
        τ_jk[b] = autoc_time_int(xs[idx])
    end

    τ_bar = mean(τ_jk)
    # standard jackknife variance formula
    var_τ = (nblocks - 1) / nblocks * sum((τ_jk .- τ_bar).^2)
    δτ = sqrt(var_τ)
    return τ_full, δτ
end

function cross_autoc_time_int(x, y)
    @assert length(x) == length(y) "streams must have equal length"
    M = length(x)
    avg = (mean(x) + mean(y)) / 2
    xz = x .- avg
    yz = y .- avg

    Γxx0 = sum(xz .* xz) / M
    Γyy0 = sum(yz .* yz) / M
    Γxy0 = sum(xz .* yz) / M
    Γ0 = Γxx0 + Γyy0 + 2Γxy0

    C = 0.0
    W = 0

    for t in 1:(M - 1)
        Γxx = Γyy = Γxy = Γyx = 0.0
        for k in 1:(M - t)
            Γxx += xz[k] * xz[k + t]
            Γyy += yz[k] * yz[k + t]
            Γxy += xz[k] * yz[k + t]
            Γyx += yz[k] * xz[k + t]
        end
        n = M - t
        Γt = (Γxx + Γyy + Γxy + Γyx) / n
        (Γt <= 0) && break
        C += Γt
        W = t
    end

    C = Γ0 + 2C
    τ = C / (2Γ0)
    δτ = τ * sqrt((4W + 2) / M) # Wolff's standard error estimate
    return τ, δτ
end

function cross_autoc_time_int_jackknife(x::Vector{<:Real}, y::Vector{<:Real}, block_size::Int)
    M = length(x)
    nblocks = M ÷ block_size
    @assert nblocks > 1 "need more than one block"

    Mtrim = nblocks * block_size
    xs = @view x[1:Mtrim]
    ys = @view y[1:Mtrim]

    # full-sample estimate
    τ_full, _ = cross_autoc_time_int(xs, ys)

    # leave-one-block-out estimates
    τ_jk = Vector{Float64}(undef, nblocks)

    for b in 1:nblocks
        idx = vcat(1:(b-1)*block_size, (b*block_size+1):Mtrim)
        τ_jk[b], _ = cross_autoc_time_int(xs[idx], ys[idx])
    end

    τ_bar = mean(τ_jk)
    # standard jackknife variance formula
    var_τ = (nblocks - 1) / nblocks * sum((τ_jk .- τ_bar).^2)
    δτ = sqrt(var_τ)

    return τ_full, δτ
end

function autoc_time_int_uw(x::Vector{<:Real})
    id = rand(Int64)
    x_uw = uwreal(x, "#$id#")
    wpm = Dict{String,Vector{Float64}}()
    wpm["#$id#"] = [-1.0, 4.0, -1.0, -1.0]
    uwerr(x_uw, wpm)
    return taui(x_uw, "#$id#"), dtaui(x_uw, "#$id#")
end

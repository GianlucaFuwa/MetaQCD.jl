struct Bootstrap{T<:Function} <: AbstractErrorEstimator
    nboot::Int64
    τ::Int64
    func::T
    function Bootstrap(; nboot=1000, τ=0, func=mean)
        @assert nboot >= 1
        @assert τ >= 0
        return new{typeof(func)}(nboot, τ, func)
    end
end

function (b::Bootstrap)(x::Vector{<:Real}, weights=nothing)
    N = length(x)

    if b.τ == 0
        τ = autoc_time_int(x)
        B = round(Int64, 2τ, RoundNearestTiesAway)
    else
        B = Int64(2b.τ)
    end

    itvl = round(Int, N / B, RoundNearestTiesAway)
    A = zeros(eltype(x), b.nboot + 1)

    if weights === nothing
        A[1] = b.func(x)
        xboot = Vector{Float64}(undef, itvl*B)

        p = Progress(b.nboot, desc = "Bootstrapping...")
        @batch threadlocal=zeros(Int64, B)::Vector{Int64} for i in 2:b.nboot+1
            r = rand(1:N, itvl)

            for l in 1:itvl
                rr = range(r[l], r[l] + B - 1)
                for ii in 1:B
                    threadlocal[ii] = mod1(rr[ii], N)
                end
                view(xboot, (l-1)*B+1:l*B) .= view(x, threadlocal)
            end

            A[i] = b.func(xboot)
            next!(p)
        end

        meanA = A[1]
        stdA = std(A)
    else
        @assert length(weights) == N "length(weights) has to be equal to length(x)"
        @info "Weighted works for means, i.e., b.func = mean"
        x .*= weights
        A[1] = sum(x) / sum(weights)

        p = Progress(b.nboot, desc="Bootstrapping...")
        @batch threadlocal=zeros(Int64, B)::Vector{Int64} for i = 2:b.nboot+1
            r = rand(1:N, itvl)
            tmpE = 0.0
            tmpD = 0.0

            for l in 1:itvl
                rr = range(r[l], r[l] + B - 1)
                for ii in 1:B
                    threadlocal[ii] = mod1(rr[ii], N)
                end
                tmpE += sum(view(x, rr))
                tmpD += sum(view(weights, rr))
            end

            A[i] = tmpE / tmpD
            next!(p)
        end

        meanA = A[1]
        stdA = std(A)
    end

    return meanA, stdA, τ
end

function bbootstrap_samplesize(b::Bootstrap, weights::Vector{<:Real})
    N = length(weights)
    B = Int64(2b.τ + 1)
    itvl = round(Int64, N / B, RoundNearestTiesAway)
    A = zeros(8, b.nboot + 1)
    A[1, 1] = sum(weights)^2 / sum(weights.^2)

    for i = 2:b.nboot+1
        r = rand(1:N, itvl)
        tmpE = 0.0
        tmpD = 0.0

        for l in 1:itvl
            q = mod1.(range(r[l], r[l] + B - 1), N)
            tmpE += sum(view(weights, q))
            tmpD += sum(view(weights, q).^2)
        end

        A[1, i] += tmpE^2 / tmpD
    end

    meanA = A[1, 1]
    stdA = std(A)
    return meanA, stdA
end

function _bbootstrap_q2fit(
    data::Vector{<:Real};
    nboot = 1000,
    τ = nothing,
)
    N = length(data)
    data = abs.(data)

    if τ === nothing
        τ = autoc_time_int(abs.(data))
        B = round(Int, 2τ + 1, RoundNearestTiesAway)
    else
        B = 2τ + 1
    end

    itvl = round(Int, N / B, RoundNearestTiesAway)
    A = Matrix{Float64}(undef, 8, nboot + 1)
    A_n0 = Matrix{Float64}(undef, 8, nboot + 1)
    @. model(x, p) = 1/sqrt(2π*p[2]^2) * exp(-(x-p[1])^2/(2*p[2]^2))
    p0 = [0.0, 1.5]
    chisq(p, d, rng) = sum((d .- model(rng, p)).^2 ./ sqrt.(d))

    qmin = minimum(data)
    qmax = maximum(data)
    qrange = qmin:1:qmax
    qrange_n0 = qrange[findall(x->x!=0, qrange)]

    binned_data = Vector{Float64}(undef, length(qrange))
    binned_data_n0 = Vector{Float64}(undef, length(qrange_n0))

    data_n0 = data[findall(x->x!=0, data)];

    for (i, q) in enumerate(qrange)
        count = length(findall(x->x==q, data))
        binned_data[i] = count
    end

    for (i, q) in enumerate(qrange_n0)
        count = length(findall(x->x==q, data_n0))
        binned_data_n0[i] = count
    end

    binned_data_n0 ./= 2sum(binned_data)
    binned_data ./= sum(binned_data)
    binned_data[2:end] ./= 2

    qfit = curve_fit(model, qrange, binned_data, 1 ./ binned_data, p0)
    qfit_n0 = curve_fit(model, qrange_n0, binned_data_n0, 1 ./ binned_data_n0, p0)

    # tmpchisq = (p, d) -> chisq(p, d, qrange)
    # tmpchisq_n0 = (p, d) -> chisq(p, d, qrange_n0)

    A[1, 1] = qfit.param[2]^2
    A_n0[1, 1] = qfit_n0.param[2]^2

    p = Progress(nboot, desc = "Bootstrapping...")
    for i = 2:nboot+1
        r = rand(1:N, itvl)
        tmp = Vector{Float64}(undef, B*itvl)

        for l in 1:itvl
            q = mod1.(range(r[l], r[l] + B - 1), N)
            tmp[(l-1)*B+1:l*B] = view(data, q)
        end

        qmin = minimum(tmp)
        qmax = maximum(tmp)
        qrange = qmin:1:qmax
        qrange_n0 = qrange[findall(x->x!=0, qrange)]

        binned_tmp = Vector{Float64}(undef, length(qrange))
        binned_tmp_n0 = Vector{Float64}(undef, length(qrange) - 1)

        tmp_n0 = tmp[findall(x->x!=0, tmp)];

        for (j, q) in enumerate(qrange)
            count = length(findall(x->x==q, tmp))
            binned_tmp[j] = count
        end

        for (j, q) in enumerate(qrange_n0)
            count = length(findall(x->x==q, tmp_n0))
            binned_tmp_n0[j] = count
        end

        binned_tmp_n0 ./= 2sum(binned_tmp)
        binned_tmp ./= sum(binned_tmp)
        binned_tmp[2:end] ./= 2

        qfit = curve_fit(model, qrange, binned_tmp, p0)
        qfit_n0 = curve_fit(model, qrange_n0, binned_tmp_n0, p0)

        # tmpchisq = (p, d) -> chisq(p, d, qrange)
        # tmpchisq_n0 = (p, d) -> chisq(p, d, qrange_n0)

        A[1, i] = qfit.param[2]^2
        A_n0[1, i] = qfit_n0.param[2]^2
        next!(p)
    end

    meanA = mean(view(A, 1, :))
    stdA = std(view(A, 1, :))
    meanA_n0 = mean(view(A_n0, 1, :))
    stdA_n0 = std(view(A_n0, 1, :))

    return (meanA, stdA), (meanA_n0, stdA_n0), τ
end

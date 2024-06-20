function Base.copy!(a::Abstractfield{CPU,T}, b::Abstractfield{CPU,T}) where {T}
    check_dims(a, b)

    @batch for site in eachindex(a)
        for μ in 1:4
            a[μ, site] = b[μ, site]
        end
    end

    return nothing
end

function identity_gauges!(u::Gaugefield{CPU,T}) where {T}
    @batch for site in eachindex(u)
        for μ in 1:4
            u[μ, site] = eye3(T)
        end
    end

    u.Sg = 0
    return nothing
end

function random_gauges!(u::Gaugefield{CPU,T}) where {T}
    for site in eachindex(u)
        for μ in 1:4
            u[μ, site] = rand_SU3(T)
        end
    end

    u.Sg = calc_gauge_action(u)
    return nothing
end

function clear!(u::Abstractfield{CPU,T}) where {T} # set all link variables to zero
    @batch for site in eachindex(u)
        for μ in 1:4
            u[μ, site] = zero3(T)
        end
    end

    return nothing
end

function normalize!(u::Gaugefield{CPU})
    @batch for site in eachindex(u)
        for μ in 1:4
            u[μ, site] = proj_onto_SU3(u[μ, site])
        end
    end

    return nothing
end

function add!(a::Abstractfield{CPU,T}, b::Abstractfield{CPU}, fac) where {T}
    check_dims(a, b)
    fac = T(fac)

    @batch for site in eachindex(a)
        for μ in 1:4
            a[μ, site] += fac * b[μ, site]
        end
    end

    return nothing
end

function mul!(a::Abstractfield{CPU,T}, α::Number) where {T}
    α = T(α)

    @batch for site in eachindex(a)
        for μ in 1:4
            a[μ, site] *= α
        end
    end

    return nothing
end

function leftmul!(a::Abstractfield{CPU}, b::Abstractfield{CPU})
    check_dims(a, b)

    @batch for site in eachindex(a)
        for μ in 1:4
            a[μ, site] = cmatmul_oo(b[μ, site], a[μ, site])
        end
    end

    return nothing
end

function leftmul_dagg!(a::Abstractfield{CPU}, b::Abstractfield{CPU})
    check_dims(a, b)

    @batch for site in eachindex(a)
        for μ in 1:4
            a[μ, site] = cmatmul_do(b[μ, site], a[μ, site])
        end
    end

    return nothing
end

function rightmul!(a::Abstractfield{CPU}, b::Abstractfield{CPU})
    check_dims(a, b)

    @batch for site in eachindex(a)
        for μ in 1:4
            a[μ, site] = cmatmul_oo(a[μ, site], b[μ, site])
        end
    end

    return nothing
end

function rightmul_dagg!(a::Abstractfield{CPU,T}, b::Abstractfield{CPU,T}) where {T}
    check_dims(a, b)

    @batch for site in eachindex(a)
        for μ in 1:4
            a[μ, site] = cmatmul_od(a[μ, site], b[μ, site])
        end
    end

    return nothing
end

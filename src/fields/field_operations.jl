function Base.copy!(a::AbstractField{CPU,T}, b::AbstractField{CPU,T}) where {T}
    check_dims(a, b)

    @batch for site in eachindex(a)
        for μ in 1:4
            a[μ, site] = b[μ, site]
        end
    end

    update_halo!(a)
    return nothing
end

function identity_gauges!(u::Gaugefield{CPU,T}) where {T}
    @batch for site in eachindex(u)
        for μ in 1:4
            u[μ, site] = eye3(T)
        end
    end

    u.Sg = 0.0
    update_halo!(u)
    return nothing
end

function random_gauges!(u::Gaugefield{CPU,T}) where {T}
    for site in eachindex(u)
        for μ in 1:4
            u[μ, site] = rand_SU3(T)
        end
    end

    u.Sg = calc_gauge_action(u)
    update_halo!(u)
    return nothing
end

function clear!(u::AbstractField{CPU,T}) where {T} # set all link variables to zero
    @batch for site in eachindex(u)
        for μ in 1:4
            u[μ, site] = zero3(T)
        end
    end

    update_halo!(u)
    return nothing
end

function normalize!(u::Gaugefield{CPU})
    @batch for site in eachindex(u)
        for μ in 1:4
            u[μ, site] = proj_onto_SU3(u[μ, site])
        end
    end

    update_halo!(u)
    return nothing
end

function LinearAlgebra.norm(u::AbstractField{CPU})
    norm2 = 0.0

    @batch reduction=(+, norm2) for site in eachindex(u)
        for μ in 1:4
            norm2 += cnorm2(u[μ, site])
        end
    end

    return distributed_reduce(norm2, +, u)
end

function add!(a::AbstractField{CPU,T}, b::AbstractField{CPU}, fac) where {T}
    check_dims(a, b)
    fac = T(fac)

    @batch for site in eachindex(a)
        for μ in 1:4
            a[μ, site] += fac * b[μ, site]
        end
    end

    update_halo!(a)
    return nothing
end

function mul!(a::AbstractField{CPU,T}, α::Number) where {T}
    α = T(α)

    @batch for site in eachindex(a)
        for μ in 1:4
            a[μ, site] *= α
        end
    end

    update_halo!(a)
    return nothing
end

function leftmul!(a::AbstractField{CPU}, b::AbstractField{CPU})
    check_dims(a, b)

    @batch for site in eachindex(a)
        for μ in 1:4
            a[μ, site] = cmatmul_oo(b[μ, site], a[μ, site])
        end
    end

    update_halo!(a)
    return nothing
end

function leftmul_dagg!(a::AbstractField{CPU}, b::AbstractField{CPU})
    check_dims(a, b)

    @batch for site in eachindex(a)
        for μ in 1:4
            a[μ, site] = cmatmul_do(b[μ, site], a[μ, site])
        end
    end

    update_halo!(a)
    return nothing
end

function rightmul!(a::AbstractField{CPU}, b::AbstractField{CPU})
    check_dims(a, b)

    @batch for site in eachindex(a)
        for μ in 1:4
            a[μ, site] = cmatmul_oo(a[μ, site], b[μ, site])
        end
    end

    update_halo!(a)
    return nothing
end

function rightmul_dagg!(a::AbstractField{CPU,T}, b::AbstractField{CPU,T}) where {T}
    check_dims(a, b)

    @batch for site in eachindex(a)
        for μ in 1:4
            a[μ, site] = cmatmul_od(a[μ, site], b[μ, site])
        end
    end

    update_halo!(a)
    return nothing
end

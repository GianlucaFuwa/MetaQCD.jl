function Base.copy!(a::AbstractField{CPU,T}, b::AbstractField{CPU,T}) where {T}
    check_dims(a, b)

    @batch for μsite in allindices(a)
        a[μsite] = b[μsite]
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    return nothing
end

function identity_gauges!(u::Gaugefield{CPU,T}) where {T}
    @batch for μsite in allindices(u)
        u[μsite] = eye3(T)
    end

    u.Sg = 0.0
    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
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
    @batch for μsite in allindices(u)
        u[μsite] = zero3(T)
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    return nothing
end

function normalize!(u::Gaugefield{CPU})
    @batch for μsite in allindices(u)
        u[μsite] = proj_onto_SU3(u[μsite])
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    return nothing
end

function LinearAlgebra.norm(u::AbstractField{CPU}, ::Val{2}) # avg 2-norm
    norm2 = 0.0

    @batch reduction=(+, norm2) for site in eachindex(u)
        for μ in 1:4
            norm2 += cnorm2(u[μ, site])
        end
    end

    norm2 /= 4u.NV
    return distributed_reduce(norm2, +, u)
end

function LinearAlgebra.norm(u::AbstractField{CPU}, ::Val{Inf}) # max of 2-norms
    normsup = 0.0

    @batch reduction=(max, normsup) for site in eachindex(u)
        for μ in 1:4
            normsup = max(normsup, cnorm2(u[μ, site])) 
        end
    end

    return distributed_reduce(normsup, max, u)
end

function add!(a::AbstractField{CPU,T}, b::AbstractField{CPU}, fac) where {T}
    check_dims(a, b)
    fac = T(fac)

    @batch for μsite in allindices(a)
        a[μsite] += fac * b[μsite]
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    # We assume that b's halo is already up-to-date before calling this
    return nothing
end

function mul!(a::AbstractField{CPU,T}, α::Number) where {T}
    α = T(α)

    @batch for μsite in allindices(a)
        a[μsite] *= α
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    return nothing
end

function leftmul!(a::AbstractField{CPU}, b::AbstractField{CPU})
    check_dims(a, b)

    @batch for μsite in allindices(a)
        a[μsite] = cmatmul_oo(b[μsite], a[μsite])
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    return nothing
end

function leftmul_dagg!(a::AbstractField{CPU}, b::AbstractField{CPU})
    check_dims(a, b)

    @batch for μsite in allindices(a)
        a[μsite] = cmatmul_do(b[μsite], a[μsite])
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    return nothing
end

function rightmul!(a::AbstractField{CPU}, b::AbstractField{CPU})
    check_dims(a, b)

    @batch for μsite in allindices(a)
        a[μsite] = cmatmul_oo(a[μsite], b[μsite])
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    return nothing
end

function rightmul_dagg!(a::AbstractField{CPU,T}, b::AbstractField{CPU,T}) where {T}
    check_dims(a, b)

    @batch for μsite in allindices(a)
        a[μsite] = cmatmul_od(a[μsite], b[μsite])
    end

    # INFO: don't need to do halo exchange here, since we iterate over all indices
    # including halo regions
    return nothing
end

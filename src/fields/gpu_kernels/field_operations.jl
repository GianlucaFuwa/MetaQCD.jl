function identity_gauges!(u::Gaugefield{B,T}) where {B<:GPU,T}
    @latmap(Sequential(), Val(1), identity_gauges_kernel!, u, T, eachindex(u))
    u.Sg = 0
    return nothing
end

@kernel function identity_gauges_kernel!(u, ::Type{T}, bulk_sites) where {T}
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    @unroll for μ in (1i32):(4i32)
        @inbounds u[μ, site] = eye3(T)
    end
end

function random_gauges!(u::Gaugefield{B,T}) where {B<:GPU,T}
    @latmap(Sequential(), Val(1), random_gauges_kernel!, u, T, eachindex(u))
    u.Sg = calc_gauge_action(u)
    return nothing
end

@kernel function random_gauges_kernel!(u, ::Type{T}, bulk_sites) where {T}
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    @unroll for μ in (1i32):(4i32)
        @inbounds u[μ, site] = rand_SU3(T)
    end
end

function Base.copy!(a::AbstractField{B,T}, b::AbstractField{B,T}) where {B<:GPU,T}
    check_dims(a, b)
    @latmap(Sequential(), Val(1), copy_kernel!, a, b, eachindex(a))
    return nothing
end

@kernel function copy_kernel!(a, @Const(b), bulk_sites)
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = b[μ, site]
    end
end

function clear!(u::AbstractField{B,T}) where {B<:GPU,T}
    @latmap(Sequential(), Val(1), clear_kernel!, u, T, eachindex(u))
end

@kernel function clear_kernel!(U, ::Type{T}, bulk_sites) where {T}
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    @unroll for μ in (1i32):(4i32)
        @inbounds U[μ, site] = zero3(T)
    end
end

function normalize!(u::AbstractField{B}) where {B<:GPU}
    @latmap(Sequential(), Val(1), normalize_kernel!, u, eachindex(u))
end

@kernel function normalize_kernel!(U, bulk_sites)
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    @unroll for μ in (1i32):(4i32)
        @inbounds U[μ, site] = proj_onto_SU3(U[μ, site])
    end
end

function norm(u::AbstractField{B}) where {B<:GPU}
    return @latsum(Sequential(), Val(1), Float64, norm_kenel!, u, eachindex(u))
end

@kernel function norm_kernel!(out, @Const(U), bulk_sites)
    # workgroup index, that we use to pass the reduced value to global "out"
    bi = @index(Group, Linear)
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    n = 0.0
    @unroll for μ in (1i32):(3i32)
        n += cnorm2(U[μ, site])
    end

    out_group = @groupreduce(+, n, 0.0)

    ti = @index(Local)
    if ti == 1
        @inbounds out[bi] = out_group
    end
end

function add!(a::AbstractField{B,T}, b::AbstractField{B,T}, fac) where {B<:GPU,T}
    check_dims(a, b)
    @latmap(Sequential(), Val(1), add_kernel!, a, b, T(fac), eachindex(a))
    return nothing
end

@kernel function add_kernel!(a, @Const(b), fac, bulk_sites)
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = a[μ, site] + fac * b[μ, site]
    end
end

function mul!(a::AbstractField{B,T}, α) where {B<:GPU,T}
    @latmap(Sequential(), Val(1), mul_kernel!, a, T(α), eachindex(a))
end

@kernel function mul_kernel!(a, α, bulk_sites)
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = α * a[μ, site]
    end
end

function leftmul!(a::AbstractField{B,T}, b::AbstractField{B,T}) where {B<:GPU,T}
    check_dims(a, b)
    @latmap(Sequential(), Val(1), leftmul_kernel!, a, b, eachindex(a))
    return nothing
end

@kernel function leftmul_kernel!(a, @Const(b), bulk_sites)
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_oo(b[μ, site], a[μ, site])
    end
end

function leftmul_dagg!(a::AbstractField{B,T}, b::AbstractField{B,T}) where {B<:GPU,T}
    check_dims(a, b)
    @latmap(Sequential(), Val(1), leftmul_dagg_kernel!, a, b, eachindex(a))
    return nothing
end

@kernel function leftmul_dagg_kernel!(a, @Const(b), bulk_sites)
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_do(b[μ, site], a[μ, site])
    end
end

function rightmul!(a::AbstractField{B,T}, b::AbstractField{B,T}) where {B<:GPU,T}
    check_dims(a, b)
    @latmap(Sequential(), Val(1), rightmul_kernel!, a, b, eachindex(a))
    return nothing
end

@kernel function rightmul_kernel!(a, @Const(b), bulk_sites)
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_oo(a[μ, site], b[μ, site])
    end
end

function rightmul_dagg!(a::AbstractField{B,T}, b::AbstractField{B,T}) where {B<:GPU,T}
    check_dims(a, b)
    @latmap(Sequential(), Val(1), rightmul_dagg_kernel!, a, b, eachindex(a))
    return nothing
end

@kernel function rightmul_dagg_kernel!(a, @Const(b), bulk_sites)
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_od(a[μ, site], b[μ, site])
    end
end

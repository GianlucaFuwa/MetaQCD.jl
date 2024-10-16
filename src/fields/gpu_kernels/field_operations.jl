function identity_gauges!(u::Gaugefield{B,T}) where {B<:GPU,T}
    @latmap(Sequential(), Val(1), identity_gauges_kernel!, u, T)
    u.Sg = 0
    return nothing
end

@kernel function identity_gauges_kernel!(u, ::Type{T}) where {T}
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds u[μ, site] = eye3(T)
    end
end

function random_gauges!(u::Gaugefield{B,T}) where {B<:GPU,T}
    @latmap(Sequential(), Val(1), random_gauges_kernel!, u, T)
    u.Sg = calc_gauge_action(u)
    return nothing
end

@kernel function random_gauges_kernel!(u, ::Type{T}) where {T}
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds u[μ, site] = rand_SU3(T)
    end
end

function Base.copy!(a::Abstractfield{B,T}, b::Abstractfield{B,T}) where {B<:GPU,T}
    check_dims(a, b)
    @latmap(Sequential(), Val(1), copy_kernel!, a, b)
    return nothing
end

@kernel function copy_kernel!(a, @Const(b))
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = b[μ, site]
    end
end

function clear!(u::Abstractfield{B,T}) where {B<:GPU,T}
    @latmap(Sequential(), Val(1), clear_kernel!, u, T)
end

@kernel function clear_kernel!(U, ::Type{T}) where {T}
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds U[μ, site] = zero3(T)
    end
end

function normalize!(u::Abstractfield{B}) where {B<:GPU}
    @latmap(Sequential(), Val(1), normalize_kernel!, u)
end

@kernel function normalize_kernel!(U)
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds U[μ, site] = proj_onto_SU3(U[μ, site])
    end
end

function norm(U::Gaugefield{B}) where {B}
    return @latsum(Sequential(), Val(1), Float64, norm_kenel!, U)
end

@kernel function norm_kernel!(out, @Const(U))
    # workgroup index, that we use to pass the reduced value to global "out"
    bi = @index(Group, Linear)
    site = @index(Global, Cartesian)

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

function add!(a::Abstractfield{B,T}, b::Abstractfield{B,T}, fac) where {B<:GPU,T}
    check_dims(a, b)
    @latmap(Sequential(), Val(1), add_kernel!, a, b, T(fac))
    return nothing
end

@kernel function add_kernel!(a, @Const(b), fac)
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = a[μ, site] + fac * b[μ, site]
    end
end

function mul!(a::Abstractfield{B,T}, α) where {B<:GPU,T}
    @latmap(Sequential(), Val(1), mul_kernel!, a, T(α))
end

@kernel function mul_kernel!(a, α)
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = α * a[μ, site]
    end
end

function leftmul!(a::Abstractfield{B,T}, b::Abstractfield{B,T}) where {B<:GPU,T}
    check_dims(a, b)
    @latmap(Sequential(), Val(1), leftmul_kernel!, a, b)
    return nothing
end

@kernel function leftmul_kernel!(a, @Const(b))
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_oo(b[μ, site], a[μ, site])
    end
end

function leftmul_dagg!(a::Abstractfield{B,T}, b::Abstractfield{B,T}) where {B<:GPU,T}
    check_dims(a, b)
    @latmap(Sequential(), Val(1), leftmul_dagg_kernel!, a, b)
    return nothing
end

@kernel function leftmul_dagg_kernel!(a, @Const(b))
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_do(b[μ, site], a[μ, site])
    end
end

function rightmul!(a::Abstractfield{B,T}, b::Abstractfield{B,T}) where {B<:GPU,T}
    check_dims(a, b)
    @latmap(Sequential(), Val(1), rightmul_kernel!, a, b)
    return nothing
end

@kernel function rightmul_kernel!(a, @Const(b))
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_oo(a[μ, site], b[μ, site])
    end
end

function rightmul_dagg!(a::Abstractfield{B,T}, b::Abstractfield{B,T}) where {B<:GPU,T}
    check_dims(a, b)
    @latmap(Sequential(), Val(1), rightmul_dagg_kernel!, a, b)
    return nothing
end

@kernel function rightmul_dagg_kernel!(a, @Const(b))
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_od(a[μ, site], b[μ, site])
    end
end

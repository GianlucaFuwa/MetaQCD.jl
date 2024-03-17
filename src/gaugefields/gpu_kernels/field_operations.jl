function identity_gauges!(u::Gaugefield{B,T}) where {B,T}
    @latmap(Sequential(), Val(1), identity_gauges_kernel!, u, T)
    return nothing
end

@kernel function identity_gauges_kernel!(u, T)
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds u[μ, site] = eye3(T)
    end
end

function random_gauges!(u::Gaugefield{B,T}) where {B,T}
    @latmap(Sequential(), Val(1), random_gauges_kernel!, u, T)
    return nothing
end

@kernel function random_gauges_kernel!(u, T)
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds u[μ, site] = rand_SU3(T)
    end
end

function Base.copy!(a::Abstractfield{B,T}, b::Abstractfield{B,T}) where {B,T}
    @assert dims(b) == dims(a)
    @latmap(Sequential(), Val(1), copy_kernel!, a, b)
    return nothing
end

@kernel function copy_kernel!(a, @Const(b))
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = b[μ, site]
    end
end

function clear!(u::Abstractfield{B,T}) where {B,T}
    @latmap(Sequential(), Val(1), clear_kernel!, u, T)
end

@kernel function clear_kernel!(U, T)
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds U[μ, site] = zero3(T)
    end
end

function normalize!(u::Abstractfield{B}) where {B}
    @latmap(Sequential(), Val(1), normalize_kernel!, u)
end

@kernel function normalize_kernel!(U)
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds U[μ, site] = proj_onto_SU3(U[μ, site])
    end
end

function add!(a::Abstractfield{B,T}, b::Abstractfield{B,T}, fac) where {B,T}
    @assert dims(b) == dims(a)
    @latmap(Sequential(), Val(1), add_kernel!, a, b, T(fac))
    return nothing
end

@kernel function add_kernel!(a, @Const(b), fac)
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = a[μ, site] + fac * b[μ, site]
    end
end

function mul!(a::Abstractfield{B,T}, α) where {B,T}
    @latmap(Sequential(), Val(1), mul_kernel!, a, T(α))
end

@kernel function mul_kernel!(a, α)
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = α * a[μ, site]
    end
end

function leftmul!(a::Abstractfield{B,T}, b::Abstractfield{B,T}) where {B,T}
    @assert dims(b) == dims(a)
    @latmap(Sequential(), Val(1), leftmul_kernel!, a, b)
    return nothing
end

@kernel function leftmul_kernel!(a, @Const(b))
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_oo(b[μ, site], a[μ, site])
    end
end

function leftmul_dagg!(a::Abstractfield{B,T}, b::Abstractfield{B,T}) where {B,T}
    @assert dims(b) == dims(a)
    @latmap(Sequential(), Val(1), leftmul_dagg_kernel!, a, b)
    return nothing
end

@kernel function leftmul_dagg_kernel!(a, @Const(b))
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_do(b[μ, site], a[μ, site])
    end
end

function rightmul!(a::Abstractfield{B,T}, b::Abstractfield{B,T}) where {B,T}
    @assert dims(b) == dims(a)
    @latmap(Sequential(), Val(1), rightmul_kernel!, a, b)
    return nothing
end

@kernel function rightmul_kernel!(a, @Const(b))
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_oo(a[μ, site], b[μ, site])
    end
end

function rightmul_dagg!(a::Abstractfield{B,T}, b::Abstractfield{B,T}) where {B,T}
    @assert dims(b) == dims(a)
    @latmap(Sequential(), Val(1), rightmul_dagg_kernel!, a, b)
    return nothing
end

@kernel function rightmul_dagg_kernel!(a, @Const(b))
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_od(a[μ, site], b[μ, site])
    end
end

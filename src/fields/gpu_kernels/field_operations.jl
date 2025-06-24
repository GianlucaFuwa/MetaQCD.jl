function identity_gauges!(u::Gaugefield{B,T}) where {B<:GPU,T}
    @latmap(eachindex(u), identity_gauges_gpu!, u, T)
    return nothing
end

@kernel cpu=false function identity_gauges_gpu!(u, ::Type{T}, bulk) where {T}
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in (1i32):(4i32)
        @inbounds u[μ, site] = eye3(T)
    end
end

function random_gauges!(u::Gaugefield{B,T}) where {B<:GPU,T}
    @latmap(eachindex(u), random_gauges_gpu!, u, T)
    return nothing
end

@kernel cpu=false function random_gauges_gpu!(u, ::Type{T}, bulk) where {T}
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in (1i32):(4i32)
        @inbounds u[μ, site] = rand_SU3(T)
    end
end

function Base.copy!(a::AbstractField{B,T}, b::AbstractField{B,T}) where {B<:GPU,T}
    @latmap(eachindex(a, b), copy_gpu!, a, b)
    return nothing
end

@kernel cpu=false function copy_gpu!(a, @Const(b), bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = b[μ, site]
    end
end

function clear!(u::AbstractField{B,T}) where {B<:GPU,T}
    @latmap(eachindex(u), clear_gpu!, u, T)
end

@kernel cpu=false function clear_gpu!(U, ::Type{T}, bulk) where {T}
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in (1i32):(4i32)
        @inbounds U[μ, site] = zero3(T)
    end
end

function normalize!(u::AbstractField{B}) where {B<:GPU}
    @latmap(eachindex(u), normalize_gpu!, u)
end

@kernel cpu=false function normalize_gpu!(U, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in (1i32):(4i32)
        @inbounds U[μ, site] = proj_onto_SU3(U[μ, site])
    end
end

function LinearAlgebra.norm(u::AbstractField{B}, ::Val{2}) where {B<:GPU}
    return @latsum(eachindex(u), Float64, norm2_gpu!, u)
end

@kernel cpu=false function norm2_gpu!(out, @Const(U), bulk)
    # workgroup index, that we use to pass the reduced value to global "out"
    iblock = @index(Group, Linear)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    n = 0.0
    @unroll for μ in (1i32):(4i32)
        Uₙ = U[μ, site]
        n += real(dot(Uₙ, Uₙ))
    end

    out_group = @groupreduce(+, n, 0.0)

    ithread = @index(Local)
    if ithread == 1
        @inbounds out[iblock] = out_group
    end
end

function LinearAlgebra.norm(u::AbstractField{B}, ::Val{Inf}) where {B<:GPU}
    return @latsup(eachindex(u), Float64, norminf_gpu!, u)
end

@kernel cpu=false function norminf_gpu!(out, @Const(U), bulk)
    # workgroup index, that we use to pass the reduced value to global "out"
    iblock = @index(Group, Linear)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    normsup = 0.0
    @unroll for μ in (1i32):(4i32)
        Uₙ = U[μ, site]
        normsup = max(normsup, real(dot(Uₙ, Uₙ)))
    end

    out_group = @groupreduce(max, normsup, 0.0)

    ithread = @index(Local)
    if ithread == 1
        @inbounds out[iblock] = out_group
    end
end

function add!(a::AbstractField{B,T}, b::AbstractField{B,T}, fac) where {B<:GPU,T}
    @latmap(eachindex(a, b), add_gpu!, a, b, T(fac))
    return nothing
end

@kernel cpu=false function add_gpu!(a, @Const(b), fac, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = a[μ, site] + fac * b[μ, site]
    end
end

function mul!(u::AbstractField{B,T}, α) where {B<:GPU,T}
    @latmap(eachindex(u), mul_gpu!, u, T(α))
end

@kernel cpu=false function mul_gpu!(a, α, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = α * a[μ, site]
    end
end

function leftmul!(a::AbstractField{B,T}, b::AbstractField{B,T}) where {B<:GPU,T}
    @latmap(eachindex(a, b), leftmul_gpu!, a, b)
    return nothing
end

@kernel cpu=false function leftmul_gpu!(a, @Const(b), bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_oo(b[μ, site], a[μ, site])
    end
end

function leftmul_dagg!(a::AbstractField{B,T}, b::AbstractField{B,T}) where {B<:GPU,T}
    @latmap(eachindex(a, b), leftmul_dagg_gpu!, a, b)
    return nothing
end

@kernel cpu=false function leftmul_dagg_gpu!(a, @Const(b), bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_do(b[μ, site], a[μ, site])
    end
end

function rightmul!(a::AbstractField{B,T}, b::AbstractField{B,T}) where {B<:GPU,T}
    @latmap(eachindex(a, b), rightmul_gpu!, a, b)
    return nothing
end

@kernel cpu=false function rightmul_gpu!(a, @Const(b), bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_oo(a[μ, site], b[μ, site])
    end
end

function rightmul_dagg!(a::AbstractField{B,T}, b::AbstractField{B,T}) where {B<:GPU,T}
    @latmap(eachindex(a, b), rightmul_dagg_gpu!, a, b)
    return nothing
end

@kernel cpu=false function rightmul_dagg_gpu!(a, @Const(b), bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in (1i32):(4i32)
        @inbounds a[μ, site] = cmatmul_od(a[μ, site], b[μ, site])
    end
end

function Base.deepcopy(u::AbstractField{B,T,M}) where {B,T,M}
    ucopy = similar(u)
    copy!(ucopy, u)
    return ucopy
end

function Base.copy!(a::AbstractField{B,T}, b::AbstractField{B,T}) where {B,T}
    parallelfor(allindices(a, b), B, Val(false), (), (a,), (a, b)) do μsite, (a, b)
        a[μsite] = b[μsite]
    end

    return nothing
end

function identity_gauges!(u::Gaugefield{B,T,M}) where {B,T,M}
    parallelfor(allindices(u), B, Val(M), (), (u,), (u,)) do μsite, (u,)
        u[μsite] = eye3(T)
    end

    return nothing
end

function random_gauges!(u::Gaugefield{B,T,M}) where {B,T,M}
    parallelfor(allindices(u), B, Val(M), (), (u,), (u,)) do μsite, (u,)
        u[μsite] = rand_SU3(T)
    end

    return nothing
end

function clear!(u::AbstractField{B,T,M}) where {B,T,M} # set all link variables to zero
    parallelfor(allindices(u), B, Val(M), (), (u,), (u,)) do μsite, (u,)
        u[μsite] = zero(u[μsite])
    end

    return nothing
end

function normalize!(u::Gaugefield{B,T,M}) where {B,T,M}
    parallelfor(allindices(u), B, Val(M), (), (u,), (u,)) do μsite, (u,)
        u[μsite] = proj_onto_SU3(u[μsite])
    end

    return nothing
end

function LinearAlgebra.tr(u::GaugeLikeField{B,T,M}) where {B,T,M}
    trace = parallelfor_sum(allindices(u), 0.0, B, Val(M), (), (), (u,)) do t, μsite, (u,)
        t = tr(u[μsite])
    end

    trace /= 4length(u)
    return distributed_reduce(trace, +, u)
end

function LinearAlgebra.norm(u::GaugeLikeField{B,T,M}, ::Val{2}) where {B,T,M}# avg 2-norm
    norm2 = parallelfor_sum(allindices(u), 0.0, B, Val(M), (), (), (u,)) do n2, μsite, (U,)
        n2 += norm(U[μsite], 2)
    end

    norm2 /= 4length(u)
    return distributed_reduce(norm2, +, u)
end

function LinearAlgebra.norm(u::GaugeLikeField{B,T,M}, ::Val{Inf}) where {B,T,M}
    normsup = parallelfor_max(allindices(u), typemin(Float64), B, (u,)) do nsup, μsite, (u,)
        nsup = max(nsup, norm(u[μsite], 2)) 
    end

    return distributed_reduce(normsup, max, u)
end

function add!(a::AbstractField{B,T}, b::AbstractField{B,T}, fac) where {B,T}
    parallelfor(allindices(a, b), B, Val(false), (), (a,), (a, b)) do μsite, (a, b)
        a[μsite] += T(fac) * b[μsite]
    end

    return nothing
end

function mul!(a::AbstractField{B,T,M}, α::Number) where {B,T,M}
    parallelfor(allindices(a), B, Val(M), (), (a,), (a,)) do μsite, (a,)
        a[μsite] *= T(α)
    end

    return nothing
end

function leftmul!(a::GaugeLikeField{B,T}, b::GaugeLikeField{B,T}) where {B,T}
    parallelfor(allindices(a, b), B, Val(false), (), (a,), (a, b)) do μsite, (a, b)
        a[μsite] = cmatmul_oo(b[μsite], a[μsite])
    end

    return nothing
end

function leftmul_dagg!(a::GaugeLikeField{B,T}, b::GaugeLikeField{B,T}) where {B,T}
    parallelfor(allindices(a, b), B, Val(false), (), (a,), (a, b)) do μsite, (a, b)
        a[μsite] = cmatmul_do(b[μsite], a[μsite])
    end

    return nothing
end

function rightmul!(a::GaugeLikeField{B,T}, b::GaugeLikeField{B,T}) where {B,T}
    parallelfor(allindices(a, b), B, Val(false), (), (a,), (a, b)) do μsite, (a, b)
        a[μsite] = cmatmul_oo(a[μsite], b[μsite])
    end

    return nothing
end

function rightmul_dagg!(a::GaugeLikeField{B,T}, b::GaugeLikeField{B,T}) where {B,T}
    parallelfor(allindices(a, b), B, Val(false), (), (a,), (a, b)) do μsite, (a, b)
        a[μsite] = cmatmul_od(a[μsite], b[μsite])
    end

    return nothing
end

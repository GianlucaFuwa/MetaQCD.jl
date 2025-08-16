function Base.deepcopy(u::AbstractField{B,T,M}) where {B,T,M}
    ucopy = similar(u)
    copy!(ucopy, u)
    return ucopy
end

function Base.copy!(a::TF, b::TF) where {B,T,M,TF<:AbstractField{B,T,M}}
    parallelfor(allindices(a, b), B, Val(M), (), (a,), (a, b)) do μsite, (a, b)
        @inbounds a[μsite] = b[μsite]
    end

    return nothing
end

function identity_gauges!(u::Gaugefield{B,T,M}) where {B,T,M}
    parallelfor(eachindex(u), B, Val(M), (), (u,), (u,)) do site, (u,)
        u[1, site] = eye3(T)
        u[2, site] = eye3(T)
        u[3, site] = eye3(T)
        u[4, site] = eye3(T)
    end

    return nothing
end

function random_gauges!(u::Gaugefield{B,T,M}) where {B,T,M}
    parallelfor(eachindex(u), B, Val(M), (), (u,), (u,)) do site, (u,)
        u[1, site] = rand_SU3(T)
        u[2, site] = rand_SU3(T)
        u[3, site] = rand_SU3(T)
        u[4, site] = rand_SU3(T)
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
    parallelfor(eachindex(u), B, Val(M), (), (u,), (u,)) do site, (u,)
        u[1, site] = proj_onto_SU3(u[1, site])
        u[2, site] = proj_onto_SU3(u[2, site])
        u[3, site] = proj_onto_SU3(u[3, site])
        u[4, site] = proj_onto_SU3(u[4, site])
    end

    return nothing
end

function LinearAlgebra.tr(u::GaugeLikeField{B,T,M}) where {B,T,M}
    trace = parallelfor_sum(eachindex(u), 0.0, B, Val(M), (), (), (u,)) do t, site, (u,)
        t = tr(u[1, site])
        t += tr(u[2, site])
        t += tr(u[3, site])
        t += tr(u[4, site])
    end

    trace /= 4length(u)
    return distributed_reduce(trace, +, u)
end

function LinearAlgebra.norm(u::GaugeLikeField{B,T,M}, ::Val{2}) where {B,T,M}# avg 2-norm
    norm2 = parallelfor_sum(eachindex(u), 0.0, B, Val(M), (), (), (u,)) do n2, site, (U,)
        n2 = norm(U[1, site], 2)
        n2 += norm(U[2, site], 2)
        n2 += norm(U[3, site], 2)
        n2 += norm(U[4, site], 2)
    end

    norm2 /= 4length(u)
    return distributed_reduce(norm2, +, u)
end

function LinearAlgebra.norm(u::GaugeLikeField{B,T,M}, ::Val{Inf}) where {B,T,M}
    normsup = parallelfor_max(eachindex(u), typemin(Float64), B, (u,)) do nsup, site, (u,)
        nsup = max(nsup, norm(u[1, site], 2)) 
        nsup = max(nsup, norm(u[2, site], 2)) 
        nsup = max(nsup, norm(u[3, site], 2)) 
        nsup = max(nsup, norm(u[4, site], 2)) 
    end

    return distributed_reduce(normsup, max, u)
end

function add!(a::TF, b::TF, fac) where {B,T,M,TF<:AbstractField{B,T,M}}
    parallelfor(allindices(a, b), B, Val(M), (), (a,), (a, b)) do μsite, (a, b)
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

function leftmul!(a::GaugeLikeField{B,T,M}, b::GaugeLikeField{B,T,M}) where {B,T,M}
    parallelfor(eachindex(a, b), B, Val(M), (), (a,), (a, b)) do site, (a, b)
        a[1, site] = cmatmul_oo(b[1, site], a[1, site])
        a[2, site] = cmatmul_oo(b[2, site], a[2, site])
        a[3, site] = cmatmul_oo(b[3, site], a[3, site])
        a[4, site] = cmatmul_oo(b[4, site], a[4, site])
    end

    return nothing
end

function leftmul_dagg!(a::GaugeLikeField{B,T,M}, b::GaugeLikeField{B,T,M}) where {B,T,M}
    parallelfor(eachindex(a, b), B, Val(M), (), (a,), (a, b)) do site, (a, b)
        a[1, site] = cmatmul_do(b[1, site], a[1, site])
        a[2, site] = cmatmul_do(b[2, site], a[2, site])
        a[3, site] = cmatmul_do(b[3, site], a[3, site])
        a[4, site] = cmatmul_do(b[4, site], a[4, site])
    end

    return nothing
end

function rightmul!(a::GaugeLikeField{B,T,M}, b::GaugeLikeField{B,T,M}) where {B,T,M}
    parallelfor(eachindex(a, b), B, Val(M), (), (a,), (a, b)) do site, (a, b)
        a[1, site] = cmatmul_oo(a[1, site], b[1, site])
        a[2, site] = cmatmul_oo(a[2, site], b[2, site])
        a[3, site] = cmatmul_oo(a[3, site], b[3, site])
        a[4, site] = cmatmul_oo(a[4, site], b[4, site])
    end

    return nothing
end

function rightmul_dagg!(a::GaugeLikeField{B,T,M}, b::GaugeLikeField{B,T,M}) where {B,T,M}
    parallelfor(eachindex(a, b), B, Val(M), (), (a,), (a, b)) do site, (a, b)
        a[1, site] = cmatmul_od(a[1, site], b[1, site])
        a[2, site] = cmatmul_od(a[2, site], b[2, site])
        a[3, site] = cmatmul_od(a[3, site], b[3, site])
        a[4, site] = cmatmul_od(a[4, site], b[4, site])
    end

    return nothing
end

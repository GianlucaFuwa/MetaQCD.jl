function Base.deepcopy(u::AbstractField{B,T,M}) where {B,T,M}
    ucopy = similar(u)
    copy!(ucopy, u)
    return ucopy
end

function Base.copy!(a::AbstractField{B,T,M}, b::AbstractField{B,T,M}) where {B,T,M}
    parallelfor(allindices(a, b), B, Val(M), (), (a,), (a, b)) do μsite, a, b
        a[μsite] = b[μsite]
    end

    return nothing
end

function identity_gauges!(u::Gaugefield{B,T,M}) where {B,T,M}
    parallelfor(allindices(u), B, Val(M), (), (u,), (u,)) do μsite, u
        u[μsite] = eye3(T)
    end

    return nothing
end

function random_gauges!(u::Gaugefield{B,T,M}) where {B,T,M}
    parallelfor(allindices(u), B, Val(M), (), (u,), (u,)) do μsite, u
        u[μsite] = rand_SU3(T)
    end

    return nothing
end

function clear!(u::AbstractField{B,T,M}) where {B,T,M} # set all link variables to zero
    parallelfor(allindices(u), B, Val(M), (), (u,), (u,)) do μsite, u
        u[μsite] = zero(u[μsite])
    end

    return nothing
end

function normalize!(u::Gaugefield{B,T,M}) where {B,T,M}
    parallelfor(allindices(u), B, Val(M), (), (u,), (u,)) do μsite, u
        u[μsite] = proj_onto_SU3(u[μsite])
    end

    return nothing
end

function LinearAlgebra.tr(u::AbstractField{B,T,M}) where {B,T,M}
    trace = parallelfor_sum(eachindex(u), 0.0, B, Val(M), (), (), (u,)) do t, site, u
        for μ in 1:4
            t += tr(u[μ, site])
        end
        t
    end

    trace /= 4length(u)
    return distributed_reduce(trace, +, u)
end

function LinearAlgebra.norm(U::AbstractField{B,T,M}, ::Val{2}) where {B,T,M}# avg 2-norm
    norm2 = parallelfor_sum(eachindex(U), 0.0, B, Val(M), (), (), (U,)) do n2, site, U
        for μ in 1:4
            n2 += norm(U[μ, site], 2)
        end
        n2
    end

    norm2 /= 4length(U)
    return distributed_reduce(norm2, +, U)
end

function LinearAlgebra.norm(u::AbstractField{B,T,M}, ::Val{Inf}) where {B,T,M}
    normsup = parallelfor_max(allindices(u), typemin(Float64), B) do nsup, μsite
        nsup = max(nsup, norm(u[μsite], 2)) 
    end

    return distributed_reduce(normsup, max, u)
end

function add!(a::AbstractField{B,T,M}, b::AbstractField{B,T,M}, fac) where {B,T,M}
    parallelfor(allindices(a, b), B, Val(M), (), (a,), (a, b)) do μsite, a, b
        a[μsite] += fac * b[μsite]
    end

    return nothing
end

function mul!(a::AbstractField{B,T,M}, α::Number) where {B,T,M}
    α = T(α)

    parallelfor(allindices(a), B, Val(M), (), (a,), (a,)) do μsite, a
        a[μsite] *= α
    end

    return nothing
end

function leftmul!(a::AbstractField{B,T,M}, b::AbstractField{B,T,M}) where {B,T,M}
    parallelfor(allindices(a, b), B, Val(M), (), (a,), (a, b)) do μsite, a, b
        a[μsite] = cmatmul_oo(b[μsite], a[μsite])
    end

    return nothing
end

function leftmul_dagg!(a::AbstractField{B,T,M}, b::AbstractField{B,T,M}) where {B,T,M}
    parallelfor(allindices(a, b), B, Val(M), (), (a,), (a, b)) do μsite, a, b
        a[μsite] = cmatmul_do(b[μsite], a[μsite])
    end

    return nothing
end

function rightmul!(a::AbstractField{B,T,M}, b::AbstractField{B,T,M}) where {B,T,M}
    parallelfor(allindices(a, b), B, Val(M), (), (a,), (a, b)) do μsite, a, b
        a[μsite] = cmatmul_oo(a[μsite], b[μsite])
    end

    return nothing
end

function rightmul_dagg!(a::AbstractField{B,T,M}, b::AbstractField{B,T,M}) where {B,T,M}
    parallelfor(allindices(a, b), B, Val(M), (), (a,), (a, b)) do μsite, a, b
        a[μsite] = cmatmul_od(a[μsite], b[μsite])
    end

    return nothing
end

function Base.copy!(a::T, b::T) where {B,T<:AbstractField{B}}
    parallelfor(allindices(a, b), B) do μsite
        a[μsite] = b[μsite]
    end

    return nothing
end

function identity_gauges!(u::Gaugefield{B,T}) where {B,T}
    parallelfor(allindices(u), B) do μsite
        u[μsite] = eye3(T)
    end

    return nothing
end

function random_gauges!(u::Gaugefield{B,T}) where {B,T}
    parallelfor(allindices(u), B) do μsite
        u[μsite] = rand_SU3(T)
    end

    return nothing
end

function clear!(u::AbstractField{B,T}) where {B,T} # set all link variables to zero
    parallelfor(allindices(u), B) do μsite
        u[μsite] = zero(u[μsite])
    end

    return nothing
end

function normalize!(u::Gaugefield{B}) where {B}
    parallelfor(allindices(u), B) do μsite
        u[μsite] = proj_onto_SU3(u[μsite])
    end

    return nothing
end

function LinearAlgebra.norm(u::AbstractField{B}, ::Val{2}) where {B}# avg 2-norm
    norm2 = parallelfor_sum(allindices(u), 0.0, B) do n2, μsite
        n2 += cnorm2(u[μsite])
    end

    norm2 /= 4length(u)
    return distributed_reduce(norm2, +, u)
end

function LinearAlgebra.norm(u::AbstractField{B}, ::Val{Inf}) where {B}# FIXME: max of 2-norms, Inf misleading
    normsup = parallelfor_max(allindices(u), typemin(Float64), B) do nsup, μsite
        nsup = max(nsup, cnorm2(u[μsite])) 
    end

    return distributed_reduce(normsup, max, u)
end

function add!(a::T, b::T, fac) where {B,T<:AbstractField{B}}
    parallelfor(allindices(a, b), B) do μsite
        a[μsite] += fac * b[μsite]
    end

    return nothing
end

function mul!(a::AbstractField{B,T}, α::Number) where {B,T}
    α = T(α)

    parallelfor(allindices(a), B) do μsite
        a[μsite] *= α
    end

    return nothing
end

function leftmul!(a::AbstractField{B}, b::AbstractField{B}) where {B}
    parallelfor(allindices(a, b), B) do μsite
        a[μsite] = cmatmul_oo(b[μsite], a[μsite])
    end

    return nothing
end

function leftmul_dagg!(a::AbstractField{B}, b::AbstractField{B}) where {B}
    parallelfor(allindices(a, b), B) do μsite
        a[μsite] = cmatmul_do(b[μsite], a[μsite])
    end

    return nothing
end

function rightmul!(a::AbstractField{B}, b::AbstractField{B}) where {B}
    parallelfor(allindices(a, b), B) do μsite
        a[μsite] = cmatmul_oo(a[μsite], b[μsite])
    end

    return nothing
end

function rightmul_dagg!(a::AbstractField{B}, b::AbstractField{B}) where {B}
    parallelfor(allindices(a, b), B) do μsite
        a[μsite] = cmatmul_od(a[μsite], b[μsite])
    end

    return nothing
end

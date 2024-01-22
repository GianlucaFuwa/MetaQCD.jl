using BenchmarkTools
using ChainRulesCore
using Cthulhu
using MetaQCD.Utils
using LinearAlgebra
using LoopVectorization
using StaticArrays
using Zygote

struct Daggered{T}
    val::T
end

struct GaugeLink{T}
    val::T
    is_dagg::Bool
    GaugeLink(mat::T) where {T} = new{T}(mat, false)
    GaugeLink(mat::T, is_dagg::Bool) where {T} = new{T}(mat, is_dagg)
end

function Daggered(mat::T) where {T}
    return Daggered{T}(mat)
end
Base.length(d::Daggered{T}) where {T} = length(d.val)
Base.iterate(d::Daggered{T}) where {T} = iterate(d.val)
Base.iterate(d::Daggered{T}, t::Tuple{SOneTo{N}, Int64}) where {T,N} = iterate(d.val,t)

@generated function *ᶜ(a::A, b::B) where {A,B}
    if A<:Daggered
        sufa = "d"
        aval = :(a.val)
    else
        sufa = "o"
        aval = :a
    end
    if B<:Daggered
        sufb = "d"
        bval = :(b.val)
    else
        sufb = "o"
        bval = :b
    end

    fun = Symbol("cmatmul_$(sufa)$(sufb)")
    quote
        $(Expr(:meta, :inline))
        $fun($aval, $bval)
    end
end

@generated function *ᶜ(args::Vararg{Union{Daggered{T},T}, N}) where {N,T}
    if N == 0
        return :(error("Need at least 1 matrix in matmul"))
    elseif N == 1
        return :(args[1])
    else
        return quote
            $(Expr(:meta, :inline))
            tmp = Utils.eye3
            Base.Cartesian.@nexprs $N i -> tmp = tmp *ᶜ args[i]
            tmp
        end
    end
end

# @generated function remultr(args::Vararg{Union{Daggered{T},T}, N}) where {N,T}
#     quote
#         $(Expr(:meta, :inline))
#         tmp = *ᶜ(args...)
#         real(tr(tmp))
#     end
# end

@inline remultr(args...) = real(tr(*ᶜ(args...)))

@inline function Base.circshift(shift::Integer, args::Vararg{Union{Daggered{T},T}, N}) where {N,T}
    j = mod1(shift, N)
    ntuple(k -> args[k-j+ifelse(k>j,0,N)], Val(N))
end

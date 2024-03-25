module Utils

using Accessors: @set
using Base.Threads: @threads, nthreads, threadid
using LinearAlgebra
using LoopVectorization
using MuladdMacro: @muladd
using Polyester
using Random
using StaticArrays

export exp_iQ, exp_iQ_coeffs, exp_iQ_su3, get_B₁, get_B₂, get_Q, get_Q²
export gen_SU3_matrix, is_special_unitary, is_traceless_antihermitian
export kenney_laub, proj_onto_SU3, multr
export make_submatrix_12, make_submatrix_13, make_submatrix_23
export embed_into_SU3_12, embed_into_SU3_13, embed_into_SU3_23
export antihermitian, hermitian, traceless_antihermitian, traceless_hermitian
export zero2, zero3, zerov3, eye2, eye3, onev3, δ, ε_tensor, gaussian_TA_mat, rand_SU3
export SiteCoords, linear_coords, move
export Sequential, Checkerboard2, Checkerboard4
export λ, expλ
export cmatmul_oo, cmatmul_dd, cmatmul_do, cmatmul_od
export cmatmul_ooo,
    cmatmul_ood,
    cmatmul_odo,
    cmatmul_doo,
    cmatmul_odd,
    cmatmul_ddo,
    cmatmul_dod,
    cmatmul_ddd
export cmatmul_oooo,
    cmatmul_oood,
    cmatmul_oodo,
    cmatmul_odoo,
    cmatmul_dooo,
    cmatmul_oodd,
    cmatmul_oddo,
    cmatmul_ddoo,
    cmatmul_odod,
    cmatmul_dood,
    cmatmul_dodo,
    cmatmul_oddd,
    cmatmul_dddo,
    cmatmul_ddod,
    cmatmul_dodd,
    cmatmul_dddd
export cdot, cmvmul, cmvmul_d, cvmmul, cvmmul_d, ckron, ckron_sum, cmvmul_spin_proj
export _unwrap_val, SU

abstract type AbstractIterator end
struct Sequential <: AbstractIterator end
struct Checkerboard2 <: AbstractIterator end
struct Checkerboard4 <: AbstractIterator end

_unwrap_val(::Val{B}) where {B} = B

@inline eye3(::Type{T}) where {T<:AbstractFloat} = @SArray [
    one(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) one(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) one(Complex{T})
]

@inline zero3(::Type{T}) where {T<:AbstractFloat} = @SArray [
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
]

@inline eye2(::Type{T}) where {T<:AbstractFloat} = @SArray [
    one(Complex{T}) zero(Complex{T})
    zero(Complex{T}) one(Complex{T})
]

@inline zero2(::Type{T}) where {T<:AbstractFloat} = @SArray [
    zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T})
]

@inline zerov3(::Type{T}) where {T<:AbstractFloat} = @SVector [
    zero(Complex{T})
    zero(Complex{T})
    zero(Complex{T})
]

@inline onev3(::Type{T}) where {T<:AbstractFloat} = @SVector [
    one(Complex{T})
    one(Complex{T})
    one(Complex{T})
]

δ(x, y) = x == y

function ε_tensor(p::NTuple{N,Int}) where {N}
    todo = Vector{Bool}(undef, N)
    todo .= true
    first = 1
    cycles = flips = 0

    while cycles + flips < N
        first = coalesce(findnext(todo, first), 0)
        (todo[first] = !todo[first]) && return 0
        j = p[first]
        cycles += 1

        while j != first
            (todo[j] = !todo[j]) && return 0
            j = p[j]
            flips += 1
        end
    end

    return iseven(flips) ? 1 : -1
end

const SU{N,N²,T} = SMatrix{N,N,Complex{T},N²}

"""
    multr(A::SU{N,N²,T}, B::SU{N,N²,T}) where {N,N²,T}

Calculate the trace of the product of two SU(N) matrices `A` and `B` of precision `T`.
"""
@inline function multr(A::SU{N,N²,T}, B::SU{N,N²,T}) where {N,N²,T}
    # for some reason we have to convert A and B to MArrays, otherwise we get a dynamic
    # function invocation for reinterpret(...) on CUDA
    a = reinterpret(reshape, T, MMatrix(A))
    b = reinterpret(reshape, T, MMatrix(B))
    re = zero(T)
    im = zero(T)

    @turbo for i in Base.Slice(static(1):static(N)), j in Base.Slice(static(1):static(N))
        re += a[1, i, j] * b[1, j, i] - a[2, i, j] * b[2, j, i]
        im += a[1, i, j] * b[2, j, i] + a[2, i, j] * b[1, j, i]
    end

    return Complex{T}(re, im)
end

include("simd_matmul.jl")
include("simd_vecops.jl")
include("generators.jl")
include("exp.jl")
include("projections.jl")
include("sitecoords.jl")

end

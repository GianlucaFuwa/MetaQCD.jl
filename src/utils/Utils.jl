module Utils

using Accessors
using Base.Threads: @spawn, @threads, nthreads, threadid
using LinearAlgebra
using LoopVectorization
using Polyester
using Random
using StaticArrays

export exp_iQ, exp_iQ_coeffs, exp_iQ_su3, get_B₁, get_B₂, get_Q, get_Q²
export gen_SU3_matrix, is_special_unitary, is_traceless_antihermitian
export kenney_laub, proj_onto_SU3, make_submatrix, embed_into_SU3, multr
export antihermitian, hermitian, traceless_antihermitian, traceless_hermitian
export zero2, eye2, zero3, eye3, δ, ε_tensor, gaussian_su3_matrix
export SiteCoords, linear_coords, move
export Sequential, Checkerboard2, Checkerboard4
export SequentialMT, Checkerboard2MT, Checkerboard4MT
export sweep!, sweep_reduce!
export λ, expλ
export cmatmul_oo, cmatmul_dd, cmatmul_do, cmatmul_od
export cmatmul_ooo, cmatmul_ood, cmatmul_odo, cmatmul_doo, cmatmul_odd,
    cmatmul_ddo, cmatmul_dod, cmatmul_ddd
export cmatmul_oooo, cmatmul_oood, cmatmul_oodo, cmatmul_odoo, cmatmul_dooo,
    cmatmul_oodd, cmatmul_oddo, cmatmul_ddoo, cmatmul_odod, cmatmul_dood,
    cmatmul_dodo, cmatmul_oddd, cmatmul_dddo, cmatmul_ddod, cmatmul_dodd,
    cmatmul_dddd
export _unwrap_val

_unwrap_val(::Val{B}) where {B} = B

const zero3 = @SArray [
    0.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 0.0+0.0im
]

const eye3 = @SArray [
    1.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 1.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 1.0+0.0im
]

const zero2 = @SArray [
    0.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im
]

const eye2 = @SArray [
    1.0+0.0im 0.0+0.0im
    0.0+0.0im 1.0+0.0im
]

"""
Kronecker-Delta: \\
δ(x, y) = \\
{1, if x == y \\
{0, else
"""
δ(x, y) = x==y

"""
Implementation of ε-tensor from: https://github.com/JuliaMath/Combinatorics.jl
"""
function ε_tensor(p::NTuple{N, Int}) where {N}
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

@inline function multr(
    A::SMatrix{NC, NC, Complex{T}, NC2},
    B::SMatrix{NC, NC, Complex{T}, NC2},
) where {NC, NC2, T}
    a = reinterpret(reshape, T, A)
    b = reinterpret(reshape, T, B)
    re = zero(T)
    im = zero(T)

    @turbo for i ∈ Base.Slice(static(1):static(NC)), j ∈ Base.Slice(static(1):static(NC))
        re += a[1,i,j] * b[1,j,i] - a[2,i,j] * b[2,j,i]
        im += a[1,i,j] * b[2,j,i] + a[2,i,j] * b[1,j,i]
    end

    return Complex{T}(re, im)
end

include("matmul.jl")
include("generators.jl")
include("exp.jl")
include("projections.jl")
include("sitecoords.jl")
include("iterators.jl")

end

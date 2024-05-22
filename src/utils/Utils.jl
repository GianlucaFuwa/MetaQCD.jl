module Utils

using Accessors: @set
using LinearAlgebra
using LoopVectorization
using MuladdMacro: @muladd
using Polyester
using Random
using StaticArrays
using PrecompileTools: PrecompileTools

export exp_iQ, exp_iQ_coeffs, exp_iQ_su3, get_B₁, get_B₂, get_Q, get_Q²
export gen_SU3_matrix, is_special_unitary, is_traceless_antihermitian
export kenney_laub, proj_onto_SU3, multr
export make_submatrix_12, make_submatrix_13, make_submatrix_23
export embed_into_SU3_12, embed_into_SU3_13, embed_into_SU3_23
export antihermitian, hermitian, traceless_antihermitian, traceless_hermitian
export zero2, zero3, zerov3, eye2, eye3, onev3, gaussian_TA_mat, rand_SU3
export SiteCoords, eo_site, eo_site_switch, move, switch_sides
export Sequential, Checkerboard2, Checkerboard4
export λ, expλ, γ₁, γ₂, γ₃, γ₄, γ₅, σ₁₂, σ₁₃, σ₁₄, σ₂₃, σ₂₄, σ₃₄
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
export cdot, cmvmul, cmvmul_d, cvmmul, cvmmul_d
export cmvmul_color, cmvmul_d_color, cvmmul_color, cvmmul_d_color
export ckron, spintrace, cmvmul_spin_proj, spin_proj, σμν_spin_mul
export _unwrap_val, SU, restore_last_col, restore_last_row, FLOAT_TYPE

abstract type AbstractIterator end
struct Sequential <: AbstractIterator end
struct Checkerboard2 <: AbstractIterator end
struct Checkerboard4 <: AbstractIterator end

_unwrap_val(::Val{B}) where {B} = B

const FLOAT_TYPE = Dict{String,DataType}(
    "float16" => Float16,
    "half" => Float16,
    "float32" => Float32,
    "single" => Float32,
    "float64" => Float64,
    "double" => Float64,
)

@inline eye2(::Type{T}) where {T<:AbstractFloat} = @SArray [
    one(Complex{T}) zero(Complex{T})
    zero(Complex{T}) one(Complex{T})
]

@inline eye3(::Type{T}) where {T<:AbstractFloat} = @SArray [
    one(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) one(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) one(Complex{T})
]

@inline eye4(::Type{T}) where {T<:AbstractFloat} = @SArray [
    one(Complex{T}) zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) one(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) one(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T}) one(Complex{T})
]

@inline zero2(::Type{T}) where {T<:AbstractFloat} = @SArray [
    zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T})
]

@inline zero3(::Type{T}) where {T<:AbstractFloat} = @SArray [
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
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

module Utils

using Accessors: @set
using LinearAlgebra
using LoopVectorization
using MPI
using MuladdMacro: @muladd
using Polyester
using Random
using StaticArrays
using PrecompileTools: PrecompileTools

export mpi_init, mpi_comm, mpi_size, mpi_parallel, mpi_myrank, mpi_amroot, mpi_barrier
export mpi_cart_create, mpi_cart_coords, mpi_cart_shift, mpi_multirequest, mpi_send
export mpi_isend, mpi_recv, mpi_irecv!, mpi_waitall, mpi_allreduce, mpi_allgather
export mpi_bcast_isbits, mpi_write_at, update_halo!
export exp_iQ, exp_iQ_coeffs, exp_iQ_su3, get_B₁, get_B₂, get_Q, get_Q²
export gen_SU3_matrix, is_special_unitary, is_traceless_antihermitian
export kenney_laub, proj_onto_SU3, multr, cnorm2
export make_submatrix_12, make_submatrix_13, make_submatrix_23
export embed_into_SU3_12, embed_into_SU3_13, embed_into_SU3_23
export antihermitian, hermitian, traceless_antihermitian, traceless_hermitian, materialize_TA
export zero2, zero3, zerov3, eye2, eye3, onev3, gaussian_TA_mat, rand_SU3
export SiteCoords, eo_site, eo_site_switch, move, switch_sides
export cartesian_to_linear
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
export cdot, cmvmul, cmvmul_d, cvmmul, cvmmul_d, cmvmul_block
export cmvmul_color, cmvmul_d_color, cvmmul_color, cvmmul_d_color
export ckron, spintrace, cmvmul_spin_proj, spin_proj, σμν_spin_mul
export _unwrap_val, SU, restore_last_col, restore_last_row, FLOAT_TYPE
export cinv, i32, spintrace_σμν

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

struct Literal{T} end
Base.:(*)(x::Number, ::Type{Literal{T}}) where {T} = T(x)
const i32 = Literal{Int32}

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
    multr(A::SMatrix{N,N,Complex{T},N²}, B::SMatrix{N,N,Complex{T},N²}) where {N,N²,T}

Calculate the trace of the product of two complex NxN matrices `A` and `B` of precision `T`.
"""
@inline function multr(A::SU{N,N²,T}, B::SU{N,N²,T}) where {N,N²,T}
    # for some reason we have to convert A and B to MArrays, otherwise we get a dynamic
    # function invocation for reinterpret(...) on CUDA
    a = reinterpret(reshape, T, MMatrix(A))
    b = reinterpret(reshape, T, MMatrix(B))
    re = zero(Float64)
    im = zero(Float64)

    @turbo for i in Base.Slice(static(1):static(N)), j in Base.Slice(static(1):static(N))
        re += a[1, i, j] * b[1, j, i] - a[2, i, j] * b[2, j, i]
        im += a[1, i, j] * b[2, j, i] + a[2, i, j] * b[1, j, i]
    end

    return ComplexF64(re, im)
end

"""
    cnorm2(A::SMatrix{N,N,Complex{T},N²}) where {N,N²,T}

Calculate the 2-norm of the complex NxN matrix `M`
"""
@inline function cnorm2(M::SU{N,N²,T}) where {N,N²,T}
    # for some reason we have to convert A and B to MArrays, otherwise we get a dynamic
    # function invocation for reinterpret(...) on CUDA
    m = reinterpret(reshape, T, MMatrix(M))
    re = zero(Float64)

    @turbo for i in Base.Slice(static(1):static(N)), j in Base.Slice(static(1):static(N))
        re += m[1, j, i] * m[1, j, i] + m[2, j, i] * m[2, j, i]
    end

    return sqrt(re)
end

"""
    cinv(M)

Calculate the inverse of the complex matrix `M`.
"""
@inline cinv(M::SMatrix{2,2,Complex{T},4}) where {T} = inv(M)
@inline cinv(M::SMatrix{3,3,Complex{T},9}) where {T} = inv(M)
@inline cinv(M::SMatrix{4,4,Complex{T},16}) where {T} = inv(M)
# StaticArrays has speical implementations for small sizes
@inline function cinv(M::SMatrix{N,N,Complex{T},N²}) where {N,N²,T}
    Q, R = qr(M)
    S = inv_upper_tri(R)
    Minv = cmatmul_od(S, Q)
    return Minv
end

@generated function inv_upper_tri(R::SMatrix{N,N,Complex{T},N²}) where {N,N²,T}
    q = quote
        $(Expr(:meta, :inline))
        Mc = MMatrix(R)
        M = reinterpret(reshape, $T, Mc)
        v = reinterpret(reshape, $T, MVector{$N,Complex{$T}}(undef))
        @turbo for i in Base.Slice(static(1):static($N))
            den = 1 / (M[1, i, i]^2 + M[2, i, i]^2)
            v[1, i] = M[1, i, i] * den 
            v[2, i] = -M[2, i, i] * den 
        end
    end

    for k in N:-1:1
        push!(q.args, :(M[1, $k, $k] = v[1, $k]))
        push!(q.args, :(M[2, $k, $k] = v[2, $k]))
        for i in k-1:-1:1
            push!(q.args, :(Mre = zero($T)))
            push!(q.args, :(Mim = zero($T)))
            for j in i+1:k
                push!(q.args, :(Mre += M[1, $i, $j] * M[1, $j, $k] - M[2, $i, $j] * M[2, $j, $k]))
                push!(q.args, :(Mim += M[1, $i, $j] * M[2, $j, $k] + M[2, $i, $j] * M[1, $j, $k]))
            end
            push!(q.args, :(M[1, $i, $k] = -v[1, $i] * Mre + v[2, $i] * Mim))
            push!(q.args, :(M[2, $i, $k] = -v[1, $i] * Mim - v[2, $i] * Mre))
        end
    end 

    push!(q.args, :(return SMatrix(Mc)))
    return q
end

include("mpi.jl")
include("auxiliary.jl")
include("simd_matmul.jl")
include("simd_vecops.jl")
include("generators.jl")
include("exp.jl")
include("projections.jl")
include("sitecoords.jl")

end

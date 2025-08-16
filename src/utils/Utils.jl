module Utils

# INFO:
# - Cannot use functions that contain reinterpret of SArrays, e.g., multr in GPU kernels
# - Cannot use @SMatrix or @SVector in GPU kernels

using Accessors: @set
using LinearAlgebra
using LoopVectorization
using MPI
using MuladdMacro: @muladd
using Polyester
using Preferences
using Random
using StaticArrays
using StaticTools
# using PrecompileTools: PrecompileTools

export METAQCD_VERSION, to_vec
export MPI_COMM_WORLD, MPI_COMM_INSTANCE, MPI_WORLD_SIZE, MPI_INSTANCE_SIZE, MPI_INSTANCE
export MPI_NUMINSTANCES, MPI_IS_GPUAWARE, mpi_make_transferrable
export mpi_comm_instance, mpi_comm_shared, mpi_ssend, mpi_recv!, mpi_datatype, mpi_buffer
export mpi_init, mpi_comm, mpi_size, mpi_parallel, mpi_myrank, mpi_amroot, mpi_barrier
export mpi_cart_create, mpi_cart_coords, mpi_cart_shift, mpi_multirequest, mpi_send
export mpi_isend, mpi_recv, mpi_irecv!, mpi_waitall, mpi_allreduce, mpi_allgather, mpi_split
export mpi_bcast, mpi_bcast!, mpi_buffer, mpi_bcast_isbits, mpi_write_at
export PauliMatrix, exp_iQ, exp_iQ_coeffs, ExpiQCoeffs, get_B₁, get_B₂, get_Q, get_Q²
export gen_SU3_matrix, is_special_unitary, is_traceless_antihermitian
export kenney_laub, proj_onto_SU3, multr, reconstruct_su3
export make_submatrix_12, make_submatrix_13, make_submatrix_23
export embed_into_SU3_12, embed_into_SU3_13, embed_into_SU3_23
export antihermitian, hermitian, traceless_antihermitian, traceless_hermitian, materialize_TA
export zero2, zero3, zerov3, eye2, eye3, onev3, gaussian_TA_mat, rand_SU3
export SiteCoords, move, get_halo_index, map_to_half, map_to_half_switch, map_from_half
export cartesian_to_linear, linear_to_cartesian, set_ext!, switch_sides, halo_to_full
export Sequential, Checkerboard2, Checkerboard4, EvenSites, OddSites
export λ, expλ, γ1, γ2, γ3, γ4, γ5, σ12, σ13, σ14, σ23, σ24, σ34
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
export cinv, i32, spintrace_pauli, struct2dict

abstract type AbstractIterator end
struct Sequential <: AbstractIterator end
struct Checkerboard2 <: AbstractIterator end
struct Checkerboard4 <: AbstractIterator end
struct EvenSites <: AbstractIterator end
struct OddSites <: AbstractIterator end

@inline _unwrap_val(::Val{B}) where {B} = B

@inline set_ext!(::Nothing, args...) = nothing
@inline set_ext!(filename::String, args...) = filename

@inline function set_ext!(filename::StaticString{N}, ::Val{len}=Val(3)) where {N,len}
    filename[end-len-2:end-len-2] = StaticString((UInt8('0' + MPI_INSTANCE[]), 0x00))
    return filename
end

@inline function set_ext!(filename::StaticString{N}, inst, ::Val{len}=Val(3)) where {N,len}
    filename[end-len-2:end-len-2] = StaticString((UInt8('0' + inst), 0x00))
    return filename
end

const FLOAT_TYPE = Dict{String,DataType}(
    "float16" => Float16,
    "half" => Float16,
    "float32" => Float32,
    "single" => Float32,
    "float64" => Float64,
    "double" => Float64,
)

@inline function Base.convert(
    ::Type{Tout}, ::Type{SMatrix{N,M,Complex{Tin},NM}}
) where {N,M,NM,Tin,Tout<:AbstractFloat}
    return SMatrix{N,M,Complex{Tout},NM}
end

@inline function Base.convert(
    ::Type{Tout}, ::Type{SVector{N,Complex{Tin}}}
) where {N,Tin,Tout<:AbstractFloat}
    return SVector{N,Complex{Tout}}
end

struct Literal{T} end
Base.:(*)(x::Number, ::Type{Literal{T}}) where {T} = T(x)
const i32 = Literal{Int32}

function struct2dict(x::T) where {T}
    return Dict{String,Any}(string(fn) => getfield(x, fn) for fn in fieldnames(T))
end

@inline function to_vec(x::Vector, len::Int64)
    @assert length(x) == len
    return x
end

@inline to_vec(x::Number, len::Int64) = fill(x, len)
@inline to_vec(x::Tuple, len::Int64) = fill(x, len)

@inline eye2(::Type{T}) where {T<:AbstractFloat} = one(SMatrix{2,2,Complex{T},4})
@inline eye3(::Type{T}) where {T<:AbstractFloat} = one(SMatrix{3,3,Complex{T},9})
@inline eye4(::Type{T}) where {T<:AbstractFloat} = one(SMatrix{4,4,Complex{T},16})
@inline zero2(::Type{T}) where {T<:AbstractFloat} = zero(SMatrix{2,2,Complex{T},4})
@inline zero3(::Type{T}) where {T<:AbstractFloat} = zero(SMatrix{3,3,Complex{T},9})
@inline zerov3(::Type{T}) where {T<:AbstractFloat} = zero(SVector{3,Complex{T}})
@inline onev3(::Type{T}) where {T<:AbstractFloat} = SVector{3,Complex{T}}(
    (Complex{T}(1.0),Complex{T}(1.0),Complex{T}(1.0))
)

const SU{N,N²,T} = SMatrix{N,N,Complex{T},N²}

struct PauliMatrix{N,N²,T<:AbstractFloat}
    upper::SU{N,N²,T}
    lower::SU{N,N²,T}
    function PauliMatrix(λ::UniformScaling{T}, ::Val{N}) where {N,T<:AbstractFloat}
        N² = N^2
        upper = lower = zeros(SMatrix{N,N,Complex{T},N²}) + λ
        return new{N,N²,T}(upper, lower)
    end

    function PauliMatrix(upper::SU{N,N²,T}, lower::SU{N,N²,T}) where {N,N²,T<:AbstractFloat}
        return new{N,N²,T}(upper, lower)
    end
end

Base.zero(::Type{PauliMatrix{N,N²,T}}) where {N,N²,T} =
    PauliMatrix(UniformScaling(zero(T)), Val(N))
Base.one(::Type{PauliMatrix{N,N²,T}}) where {N,N²,T} =
    PauliMatrix(UniformScaling(one(T)), Val(N))
Base.eltype(::Type{PauliMatrix{N,N²,T}}) where {N,N²,T} = Complex{T}

function Base.rand(::Type{PauliMatrix{N,N²,T}}) where {N,N²,T}
    upper = hermitian(rand(SMatrix{N,N,Complex{T},N²}))
    lower = hermitian(rand(SMatrix{N,N,Complex{T},N²}))
    return PauliMatrix(upper, lower)
end

"""
    multr(A::SMatrix{N,N,Complex{T},N²}, B::SMatrix{N,N,Complex{T},N²}) where {N,N²,T}

Calculate the trace of the product of two complex NxN matrices `A` and `B` of precision `T`.
"""
@inline multr(A, B) = tr(cmatmul_oo(A, B))

"""
    cinv(M)

Calculate the inverse of the complex matrix `M`.
"""
@inline cinv(M::SMatrix{2,2,Complex{T},4}) where {T} = inv(M)
@inline cinv(M::SMatrix{3,3,Complex{T},9}) where {T} = inv(M)
@inline cinv(M::SMatrix{4,4,Complex{T},16}) where {T} = inv(M)
# StaticArrays has special implementations for small sizes
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
include("algebra.jl")
include("sitecoords.jl")

end

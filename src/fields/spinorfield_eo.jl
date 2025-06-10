"""
    even_odd(f::Spinorfield)

Create a wrapper around a `Spinorfield` to signal that it is meant to be used in the context
of even-odd preconditioning. What this amounts to is that we realign the entries such that
`ϕ -> (ϕₑ, ϕₒ)`, which is achieved by recalculating the index whenever we index into `ϕ`
or iterating only over one half of its indices.
"""
even_odd(f::Spinorfield) = SpinorfieldEO(f)

struct SpinorfieldEO{Backend,FloatType,IsDistributed,ArrayType,NumDirac} <:
    AbstractField{Backend,FloatType,IsDistributed,ArrayType}
    parent::Spinorfield{Backend,FloatType,IsDistributed,ArrayType,NumDirac}

    function SpinorfieldEO(
        f::Spinorfield{Backend,FloatType,IsDistributed,ArrayType,NumDirac}
    ) where {Backend,FloatType,IsDistributed,ArrayType,NumDirac}
        @assert iseven(f.NT) "Need even time extent for even-odd preconditioning"

        if IsDistributed
            @assert iseven(f.NT÷2) """
            Need time extent divisbile by 4 for even-odd preconditioning with \
            distributed fields
            """
        end
        
        return new{Backend,FloatType,IsDistributed,ArrayType,NumDirac}(f)
    end
end

const MPISpinorfieldEO{Backend,FloatType,ArrayType,NumDirac} =
    SpinorfieldEO{Backend,FloatType,true,ArrayType,NumDirac}

function Spinorfield(
    f::SpinorfieldEO{Backend,FloatType,IsDistributed,ArrayType,NumDirac}
) where {Backend,FloatType,IsDistributed,ArrayType,NumDirac}
    return SpinorfieldEO(f.parent)
end

is_evenodd(::SpinorfieldEO) = true # is only true for SpinorfieldEO
dims(f::SpinorfieldEO) = dims(f.parent)
local_dims(f::SpinorfieldEO) = local_dims(f.parent)
global_dims(f::SpinorfieldEO) = global_dims(f.parent)
Base.size(f::SpinorfieldEO) = size(f.parent)
Base.similar(f::SpinorfieldEO) = even_odd(Spinorfield(f.parent))
Base.eltype(::SpinorfieldEO{B,T}) where {B,T} = Complex{T}
LinearAlgebra.checksquare(f::SpinorfieldEO) = LinearAlgebra.checksquare(f.parent) ÷ 2
num_colors(::SpinorfieldEO{B,T,M,A,ND}) where {B,T,M,A,ND} = 3
num_dirac(::SpinorfieldEO{B,T,M,A,ND}) where {B,T,M,A,ND} = ND
volume(f::SpinorfieldEO) = volume(f.parent)

Base.@propagate_inbounds Base.getindex(f::SpinorfieldEO, i::Integer) = f.parent.U[i]
Base.@propagate_inbounds Base.getindex(f::SpinorfieldEO, x, y, z, t) = f.parent.U[x, y, z, t]
Base.@propagate_inbounds Base.getindex(f::SpinorfieldEO, site::SiteCoords) = f.parent.U[site]
Base.@propagate_inbounds Base.setindex!(f::SpinorfieldEO, v, i::Integer) =
    setindex!(f.parent.U, v, i)
Base.@propagate_inbounds Base.setindex!(f::SpinorfieldEO, v, x, y, z, t) =
    setindex!(f.parent.U, v, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(f::SpinorfieldEO, v, site::SiteCoords) =
    setindex!(f.parent.U, v, site)

Base.view(f::SpinorfieldEO, I::CartesianIndices{4}) = view(f.parent.U, I.indices...)
Base.view(f::SpinorfieldEO, I::Vector{CartesianIndex{4}}) = view(f.parent.U, I)

@inline function allindices(even::Bool, u::Union{Spinorfield,SpinorfieldEO})
    NX, NY, NZ, NT = size(u.U)
    @assert iseven(NT)
    last_range = even ? (1:div(NT, 2)) : (div(NT, 2)+1:NT)
    return CartesianIndices((NX, NY, NZ, last_range))
end

clear!(ϕ_eo::SpinorfieldEO) = clear!(ϕ_eo.parent)
ones!(ϕ_eo::SpinorfieldEO) = ones!(ϕ_eo.parent)

function set_source!(ϕ_eo::SpinorfieldEO{CPU,T}, site::SiteCoords, a, μ) where {T}
    ϕ = ϕ_eo.parent
    NC = num_colors(ϕ)
    ND = num_dirac(ϕ)
    @assert μ ∈ 1:ND && a ∈ 1:3
    clear!(ϕ)
    vec_index = (μ - 1) * NC + a
    tup = ntuple(i -> i == vec_index ? one(Complex{T}) : zero(Complex{T}), Val(3ND))
    _site = eo_site(site, global_dims(ϕ)..., ϕ.NV)
    ϕ[_site] = SVector{3ND,Complex{T}}(tup)
    # TODO: @hide_communication(ϕ) begin ... end
    update_halo_eo!(ϕ) # TODO: Even-odd halo exchange
    return nothing
end

function Base.copy!(ϕ_eo::TF, ψ_eo::TF) where {TF<:SpinorfieldEO{CPU}}
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    even = true

    @batch for e_site in eachindex(even, ϕ, ψ)
        ϕ[e_site] = ψ[e_site]
    end

    return nothing
end

function gaussian_pseudofermions!(ϕ_eo::SpinorfieldEO{CPU,T}) where {T}
    ϕ = ϕ_eo.parent
    sz = num_dirac(ϕ) * num_colors(ϕ)
    even = true

    for e_site in eachindex(even, ϕ)
        ϕ[e_site] = @SVector randn(Complex{T}, sz) # σ = 0.5
    end

    update_halo_eo!(ϕ)
    return nothing
end

function LinearAlgebra.mul!(ψ_eo::TF, ϕ_eo::TF, α) where {T,TF<:SpinorfieldEO{CPU,T}}
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    α = Complex{T}(α)
    even = true

    @batch for _site in eachindex(even, ϕ, ψ)
        ψ[_site] = ϕ[_site] * α
    end

    return nothing
end

function LinearAlgebra.axpy!(α, ψ_eo::TF, ϕ_eo::TF) where {T,TF<:SpinorfieldEO{CPU,T}} # even on even is the default
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    α = Complex{T}(α)
    even = true

    @batch for _site in eachindex(even, ϕ, ψ)
        ϕ[_site] += α * ψ[_site]
    end

    return nothing
end

function LinearAlgebra.axpby!(
    α, ψ_eo::TF, β, ϕ_eo::TF, even=true
) where {T,TF<:SpinorfieldEO{CPU,T}}
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    α = Complex{T}(α)
    β = Complex{T}(β)

    @batch for _site in eachindex(even, ϕ, ψ)
        ϕ[_site] = α * ψ[_site] + β * ϕ[_site]
    end

    return nothing
end

LinearAlgebra.norm(ϕ_eo::SpinorfieldEO) = sqrt(real(dot(ϕ_eo, ϕ_eo)))

function LinearAlgebra.dot(ϕ_eo::T, ψ_eo::T) where {T<:SpinorfieldEO{CPU}}
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    res = 0.0 + 0.0im # res is always double precision, even if T is single precision
    even = true

    @batch reduction = (+, res) for _site in eachindex(even, ϕ, ψ)
        res += cdot(ϕ[_site], ψ[_site])
    end

    return distributed_reduce(res, +, ϕ)
end

function dot_all(ϕ_eo::T, ψ_eo::T) where {T<:SpinorfieldEO{CPU}}
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    res = 0.0 + 0.0im # res is always double precision, even if T is single precision

    @batch reduction = (+, res) for site in eachindex(ϕ, ψ)
        res += cdot(ϕ[site], ψ[site])
    end

    return distributed_reduce(res, +, ϕ)
end

function copy_eo!(ϕ_eo::T, ψ_eo::T) where {T<:SpinorfieldEO{CPU}}
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    fdims = dims(ϕ)
    NV = ϕ.NV
    even = true

    for e_site in eachindex(even, ϕ, ψ)
        o_site = switch_sides(e_site, fdims..., NV)
        ϕ[e_site] = ψ[o_site]
    end

    return nothing
end

function copy_oe!(ϕ_eo::T, ψ_eo::T) where {T<:SpinorfieldEO{CPU}}
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    fdims = dims(ϕ)
    NV = ϕ.NV
    odd = false

    for o_site in eachindex(odd, ϕ, ψ)
        e_site = switch_sides(o_site, fdims..., NV)
        ϕ[o_site] = ψ[e_site]
    end

    return nothing
end

function axpy_oe!(α, ψ_eo::T, ϕ_eo::T) where {T<:SpinorfieldEO{CPU}}
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)
    fdims = dims(ϕ)
    NV = ϕ.NV
    even = true

    for e_site in eachindex(even, ϕ, ψ)
        o_site = switch_sides(e_site, fdims..., NV)
        ϕ[e_site] += α * ψ[o_site]
    end

    return nothing
end

function axpy_eo!(α, ψ_eo::T, ϕ_eo::T) where {T<:SpinorfieldEO{CPU}}
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)
    fdims = dims(ϕ)
    NV = ϕ.NV
    odd = false

    for o_site in eachindex(odd, ϕ, ψ)
        e_site = switch_sides(o_site, fdims..., NV)
        ϕ[o_site] += α * ψ[e_site]
    end

    return nothing
end

update_halo_eo!(::AbstractField) = nothing

function update_halo_eo!(u::AbstractMPIField)
    topology = u.topology
    comm_cart = topology.comm_cart
    comm_instance = mpi_comm_instance()
    border_sites = topology.border_sites_eo
    halo_sites = topology.halo_sites_eo

    requests = Utils.MPI.Request[]

    for dim in 1:4
        for (i, parity) in enumerate((:even, :odd))
            prev_neighbor, next_neighbor = mpi_cart_shift(comm_cart, dim-1, 1)
            prev_sites_from, next_sites_from = border_sites[parity][dim]
            prev_sites_to, next_sites_to = halo_sites[parity][dim]

            if prev_neighbor == next_neighbor == mpi_myrank(comm_instance)
                view(u, next_sites_to) .= view(u, prev_sites_from)
                view(u, prev_sites_to) .= view(u, next_sites_from)
            else
                # Use references of the links themselves as buffers
                # INFO: Here, `view` is defined such that it automatically references all four
                # directions `μ`, and we don't have to include it as an argument
                send_buf_prev = u[prev_sites_from]
                send_buf_next = u[next_sites_from]
                recv_buf_prev = u[prev_sites_to]
                recv_buf_next = u[next_sites_to]

                push!(
                    requests,
                    mpi_irecv!(recv_buf_prev, comm_cart; source=prev_neighbor, tag=1+8i),
                    mpi_isend(send_buf_next, comm_cart; dest=next_neighbor, tag=1+8i),
                    mpi_irecv!(recv_buf_next, comm_cart; source=next_neighbor, tag=2+8i),
                    mpi_isend(send_buf_prev, comm_cart; dest=prev_neighbor, tag=2+8i)
                )
            end
        end
    end

    return mpi_waitall(requests)
end

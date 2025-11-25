set_instanton!(::Gaugefield, ::Nothing) = nothing

function set_instanton!(U::Gaugefield{B,T,M}, Q::Vector{Int64}) where {B,T,M}
    set_instanton!(U, Q[MPI_INSTANCE[]+1])
    return nothing
end

function set_instanton!(U::Gaugefield{B,T,M}, Q) where {B,T,M}
    NX, NY, NZ, NT = size(U)
    xrange, yrange, zrange, trange = U.topology.bulk_sites.indices
    identity_gauges!(U)

    s = sig(T)
    s_comp = sig_comp(T)
    s_id = sig_id(T)

    field_x = 2π * abs(Q) / NX
    field_t = 2π * abs(Q) / (NX*NT)

    parallelfor(eachindex(U), B, Val(M), (), (U,), (U,)) do site, (U,)
        it = site[4]
        cit = T(cos(field_t * it))
        sit = T(sin(field_t * it))
        U[1, site] = s_comp + cit * s_id + im * sit * s
    end

    itr = CartesianIndices((xrange, yrange, zrange))

    if NT in trange
        parallelfor(itr, B, Val(M), (), (U,), (U,)) do xyz, (U,)
            ix, iy, iz = xyz.I
            cit = T(cos(field_x * ix))
            sit = T(sin(field_x * ix))
            U[4, CartesianIndex(ix, iy, iz, NT)] = s_comp + cit * s_id - im * sit * s
        end
    end

    field_y::T = Q == 0 ? 0.0 : -2π * Q / (abs(Q) * NY * NZ)
    field_z::T = Q == 0 ? 0.0 : -2π * Q / (abs(Q) * NZ)

    t = tau(T)
    t_comp = tau_comp(T)
    t_id = tau_id(T)

    parallelfor(eachindex(U), B, Val(M), (), (U,), (U,)) do site, (U,)
        iy = site[2]
        cit = T(cos(field_y * iy))
        sit = T(sin(field_y * iy))
        U[3, site] = t_comp + cit * t_id + im * sit * t
    end

    itr = CartesianIndices((xrange, zrange, trange))

    if NY in yrange
        parallelfor(itr, B, Val(M), (), (U,), (U,)) do xzt, (U,)
            ix, iz, it = xzt.I
            cit = T(cos(field_z * iz))
            sit = T(sin(field_z * iz))
            U[2, CartesianIndex(ix, NY, iz, it)] = t_comp + cit * t_id - im * sit * t
        end
    end

    normalize!(U)
    mpi_barrier(mpi_comm_instance())
    return nothing
end

@inline function sig1(::Type{T}) where {T<:AbstractFloat}
    return SMatrix{3,3,Complex{T},9}(
        one(Complex{T}), zero(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), -one(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), zero(Complex{T}), zero(Complex{T})
    )
end

@inline function sig2(::Type{T}) where {T<:AbstractFloat}
    return SMatrix{3,3,Complex{T},9}(
        one(Complex{T}), zero(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), one(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), zero(Complex{T}), zero(Complex{T})
    )
end

@inline function sig3(::Type{T}) where {T<:AbstractFloat}
    return SMatrix{3,3,Complex{T},9}(
        zero(Complex{T}), zero(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), zero(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), zero(Complex{T}), one(Complex{T})
    )
end

@inline function sig(::Type{T}) where {T<:AbstractFloat}
    return SMatrix{3,3,Complex{T},9}(
        zero(Complex{T}), zero(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), -one(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), zero(Complex{T}), zero(Complex{T})
    )
end

@inline function sig_id(::Type{T}) where {T<:AbstractFloat}
    return SMatrix{3,3,Complex{T},9}(
        one(Complex{T}), zero(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), one(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), zero(Complex{T}), zero(Complex{T})
    )
end

@inline function sig_comp(::Type{T}) where {T<:AbstractFloat}
    return SMatrix{3,3,Complex{T},9}(
        zero(Complex{T}), zero(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), zero(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), zero(Complex{T}), one(Complex{T})
    )
end

@inline function tau(::Type{T}) where {T<:AbstractFloat}
    return SMatrix{3,3,Complex{T},9}(
        one(Complex{T}), zero(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), zero(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), zero(Complex{T}), -one(Complex{T})
    )
end

@inline function tau_id(::Type{T}) where {T<:AbstractFloat}
    return SMatrix{3,3,Complex{T},9}(
        one(Complex{T}), zero(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), zero(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), zero(Complex{T}), one(Complex{T})
    )
end

@inline function tau_comp(::Type{T}) where {T<:AbstractFloat}
    return SMatrix{3,3,Complex{T},9}(
        zero(Complex{T}), zero(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), one(Complex{T}), zero(Complex{T}),
        zero(Complex{T}), zero(Complex{T}), zero(Complex{T})
    )
end

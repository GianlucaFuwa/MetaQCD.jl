set_instanton!(::Gaugefield, ::Nothing) = nothing

function set_instanton!(U::Gaugefield, Q::Vector{Int64})
    set_instanton!(U, Q[MPI_MYINSTANCE[]+1])
    return nothing
end

function set_instanton!(U::Gaugefield{B,T}, Q) where {B,T}
    NX, NY, NZ, NT = size(U)
    xrange, yrange, zrange, trange = U.topology.bulk_sites.indices
    identity_gauges!(U)

    s = sig(T)
    s_comp = sig_comp(T)
    s_id = sig_id(T)

    field_x = T(2π * abs(Q) / NX)
    field_t = T(2π * abs(Q) / (NX*NT))

    parallelfor(eachindex(U), B) do site
        it = site[4]
        cit = cos(field_t * it)
        sit = sin(field_t * it)
        U[1, site] = s_comp + cit * s_id + im * sit * s
    end

    parallelfor(CartesianIndices((xrange, yrange, zrange)), B) do xyz
        ix, iy, iz = xyz.I
        cit = cos(field_x * ix)
        sit = sin(field_x * ix)
        U[4, ix, iy, iz, NT] = s_comp + cit * s_id - im * sit * s
    end

    if Q == 0
        field_y = T(0)
        field_z = T(0)
    else
        field_y = T(-2π * Q / (abs(Q) * NY * NZ))
        field_z = T(-2π * Q / (abs(Q) * NZ))
    end

    t = tau(T)
    t_comp = tau_comp(T)
    t_id = tau_id(T)

    parallelfor(eachindex(U), B) do site
        iy = site[2]
        cit = cos(field_y * iy)
        sit = sin(field_y * iy)
        U[3, site] = t_comp + cit * t_id + im * sit * t
    end

    parallelfor(CartesianIndices((xrange, zrange, trange)), B) do xzt
        ix, iz, it = xzt.I
        cit = cos(field_z * iz)
        sit = sin(field_z * iz)
        U[2, ix, NY, iz, it] = t_comp + cit * t_id - im * sit * t
    end

    return nothing
end

@inline sig1(::Type{T}) where {T<:AbstractFloat} = @SArray [
    one(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) -one(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
]

@inline sig2(::Type{T}) where {T<:AbstractFloat} = @SArray [
    one(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) one(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
]

@inline sig3(::Type{T}) where {T<:AbstractFloat} = @SArray [
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) one(Complex{T})
]

@inline sig(::Type{T}) where {T<:AbstractFloat} = @SArray [
    one(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) -one(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
]

@inline sig_id(::Type{T}) where {T<:AbstractFloat} = @SArray [
    one(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) one(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
]

@inline sig_comp(::Type{T}) where {T<:AbstractFloat} = @SArray [
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) one(Complex{T})
]

@inline tau(::Type{T}) where {T<:AbstractFloat} = @SArray [
    one(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) -one(Complex{T})
]

@inline tau_id(::Type{T}) where {T<:AbstractFloat} = @SArray [
    one(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) one(Complex{T})
]

@inline tau_comp(::Type{T}) where {T<:AbstractFloat} = @SArray [
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
    zero(Complex{T}) one(Complex{T}) zero(Complex{T})
    zero(Complex{T}) zero(Complex{T}) zero(Complex{T})
]

function set_instanton!(U::Gaugefield{CPU,T}, Q) where {T}
    NX, NY, NZ, NT = global_dims(U)
    xrange, yrange, zrange, trange = U.topology.bulk_sites.indices
    identity_gauges!(U)

    s = sig(T)
    s_comp = sig_comp(T)
    s_id = sig_id(T)

    field_x = T(2π * abs(Q) / NX)
    field_t = T(2π * abs(Q) / (NX*NT))

    @batch for site in eachindex(U)
        it = site[4]
        cit = cos(field_t * it)
        sit = sin(field_t * it)
        U[1, site] = s_comp + cit * s_id + im * sit * s
    end

    @batch for iz in zrange
        for iy in yrange
            for ix in xrange
                cit = cos(field_x * ix)
                sit = sin(field_x * ix)
                U[4, ix, iy, iz, NT] = s_comp + cit * s_id - im * sit * s
            end
        end
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

    @batch for site in eachindex(U)
        iy = site[2]
        cit = cos(field_y * iy)
        sit = sin(field_y * iy)
        U[3, site] = t_comp + cit * t_id + im * sit * t
    end

    @batch for it in trange
        for iz in zrange
            for ix in xrange
                cit = cos(field_z * iz)
                sit = sin(field_z * iz)
                U[2, ix, NY, iz, it] = t_comp + cit * t_id - im * sit * t
            end
        end
    end

    update_halo!(U)
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

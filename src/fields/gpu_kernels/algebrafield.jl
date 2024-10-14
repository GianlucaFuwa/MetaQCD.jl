function gaussian_TA!(p::Colorfield{B,T}, ϕ) where {B,T}
    ϕ₁ = T(sqrt(1 - ϕ^2))
    ϕ₂ = T(ϕ)
    @latmap(Sequential(), Val(1), gaussian_TA_kernel!, p, ϕ₁, ϕ₂, T, eachindex(U))
end

@kernel function gaussian_TA_kernel!(P, ϕ₁, ϕ₂, ::Type{T}, bulk_sites) where {T}
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    @unroll for μ in (1i32):(4i32)
        @inbounds P[μ, site] = ϕ₁ * gaussian_TA_mat(T) + ϕ₂ * P[μ, site]
    end
end

function calc_kinetic_energy(p::Colorfield{B}) where {B}
    return @latsum(
        Sequential(), Val(1), Float64, calc_kinetic_energy_kernel!, p, eachindex(U)
    )
end

@kernel function calc_kinetic_energy_kernel!(out, @Const(P), bulk_sites)
    # workgroup index, that we use to pass the reduced value to global "out"
    bi = @index(Group, Linear)
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    k = 0.0
    @unroll for μ in (1i32):(4i32)
        @inbounds pmat = P[μ, site]
        k += real(multr(pmat, pmat))
    end

    out_group = @groupreduce(+, k, 0.0)

    ti = @index(Local)
    if ti == 1
        @inbounds out[bi] = out_group
    end
end

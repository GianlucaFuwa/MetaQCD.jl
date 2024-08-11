function gaussian_TA!(p::Colorfield{B,T}, ϕ) where {B,T}
    @latmap(Sequential(), Val(1), gaussian_TA_kernel!, p, T(sqrt(1 - ϕ^2)), T(ϕ), T)
end

@kernel function gaussian_TA_kernel!(P, ϕ₁, ϕ₂, ::Type{T}) where {T}
    site = @index(Global, Cartesian)

    @unroll for μ in (1i32):(4i32)
        @inbounds P[μ, site] = ϕ₁ * gaussian_TA_mat(T) + ϕ₂ * P[μ, site]
    end
end

function calc_kinetic_energy(p::Colorfield{B}) where {B}
    return @latsum(Sequential(), Val(1), Float64, calc_kinetic_energy_kernel!, p)
end

@kernel function calc_kinetic_energy_kernel!(out, @Const(P))
    # workgroup index, that we use to pass the reduced value to global "out"
    bi = @index(Group, Linear)
    site = @index(Global, Cartesian)

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

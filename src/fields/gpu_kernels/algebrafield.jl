function gaussian_TA!(p::Colorfield{B,T}, ϕ) where {B,T}
    ϕ₁ = T(sqrt(1 - ϕ^2))
    ϕ₂ = T(ϕ)
    @latmap(Sequential(), Val(1), gaussian_TA_gpu!, p, ϕ₁, ϕ₂, T, eachindex(p))
end

@kernel cpu=false function gaussian_TA_gpu!(P, ϕ₁, ϕ₂, ::Type{T}, bulk) where {T}
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in (1i32):(4i32)
        @inbounds P[μ, site] = ϕ₁ * gaussian_TA_mat(T) + ϕ₂ * P[μ, site]
    end
end

function calc_kinetic_energy(p::Colorfield{B}) where {B}
    return @latsum(
        Sequential(), Val(1), Float64, calc_kinetic_energy_gpu!, p, eachindex(p)
    )
end

@kernel cpu=false function calc_kinetic_energy_gpu!(out, @Const(P), bulk)
    # workgroup index, that we use to pass the reduced value to global "out"
    iblock = @index(Group, Linear)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    k = 0.0
    @unroll for μ in (1i32):(4i32)
        pmat = P[μ, site]
        k += real(tr(cmatmul_oo(pmat, pmat)))
    end

    out_group = @groupreduce(+, k, 0.0)

    ti = @index(Local)
    if ti == 1
        @inbounds out[iblock] = out_group
    end
end

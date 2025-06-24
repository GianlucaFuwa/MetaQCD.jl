function updateU!(U::Gaugefield{B,T}, Z::Colorfield{B,T}, ϵ) where {B<:GPU,T}
    @latmap(eachindex(Z, U), updateU_gf_gpu!, U, Z, T(ϵ))
    return nothing
end

@kernel cpu=false function updateU_gf_gpu!(U, @Const(Z), ϵ, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in 1i32:4i32
        @inbounds U[μ, site] = cmatmul_oo(exp(ϵ*Z[μ,site]), U[μ, site])
    end
end

function calcZ!(Z::Colorfield{B,T}, U::Gaugefield{B,T}, ϵ) where {B<:GPU,T}
    # TODO: can hide
    update_halo!(U)
    @latmap(eachindex(Z, U), calcZ_gpu!, Z, U, T(ϵ))
    return nothing
end

@kernel cpu=false function calcZ_gpu!(Z, @Const(U), ϵ, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in 1i32:4i32
        A = staple(WilsonGaugeAction(), U, μ, site)
        @inbounds AU = cmatmul_od(A, U[μ, site])
        @inbounds Z[μ, site] = ϵ * traceless_antihermitian(AU)
    end
end

function updateZ!(Z::Colorfield{B,T}, U::Gaugefield{B,T}, ϵ_old, ϵ_new, bulk) where {B<:GPU,T}
    # TODO: can hide
    update_halo!(U)
    @latmap(eachindex(Z, U), updateZ_gpu!, Z, U, T(ϵ_old), T(ϵ_new))
    return nothing
end

@kernel cpu=false function updateZ_gpu!(Z, @Const(U), ϵ_old, ϵ_new, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in 1i32:4i32
        A = staple(WilsonGaugeAction(), U, μ, site)
        @inbounds AU = cmatmul_od(A, U[μ, site])
        @inbounds Z[μ, site] = ϵ_old * Z[μ, site] + ϵ_new * traceless_antihermitian(AU)
    end
end

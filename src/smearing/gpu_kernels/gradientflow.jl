function updateU!(U::Gaugefield{B,T}, Z::Temporaryfield{B}, ϵ) where {B<:GPU,T}
    @assert dims(Z) == dims(U)
    @latmap(Sequential(), Val(1), updateU_gf_kernel!, U, Z, T(ϵ))
    return nothing
end

@kernel function updateU_gf_kernel!(U, @Const(Z), ϵ)
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
        @inbounds U[μ,site] = cmatmul_oo(exp_iQ(-im * ϵ * Z[μ,site]), U[μ,site])
    end
end

function calcZ!(Z::Temporaryfield{B,T}, U::Gaugefield{B}, ϵ) where {B<:GPU,T}
    @assert dims(Z) == dims(U)
    @latmap(Sequential(), Val(1), calcZ_kernel!, Z, U, T(ϵ))
    return nothing
end

@kernel function calcZ_kernel!(Z, @Const(U), ϵ)
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
        A = staple(WilsonGaugeAction(), U, μ, site)
        @inbounds AU = cmatmul_od(A, U[μ,site])
        @inbounds Z[μ,site] = ϵ * traceless_antihermitian(AU)
    end
end

function updateZ!(Z::Temporaryfield{B,T}, U::Gaugefield{B}, ϵ_old, ϵ_new) where {B<:GPU,T}
    @assert dims(Z) == dims(U)
    @latmap(Sequential(), Val(1), updateZ_kernel!, Z, U, T(ϵ_old), T(ϵ_new))
    return nothing
end

@kernel function updateZ_kernel!(Z, @Const(U), ϵ_old, ϵ_new)
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
        A = staple(WilsonGaugeAction(), U, μ, site)
        @inbounds AU = cmatmul_od(A, U[μ,site])
        @inbounds Z[μ,site] = ϵ_old * Z[μ,site] + ϵ_new * traceless_antihermitian(AU)
    end
end

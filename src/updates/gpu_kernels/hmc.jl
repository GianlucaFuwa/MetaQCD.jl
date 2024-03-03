function updateU!(U::Gaugefield{B,T}, hmc::HMC, fac) where {B<:GPU,T}
    ϵ = hmc.Δτ * fac
    P = hmc.P
    @assert dims(U) == dims(P)
    @latmap(Sequential(), Val(1), updateU_kernel!, U, P, convert(T, ϵ))
    return nothing
end

@kernel function updateU_kernel!(U, @Const(P), ϵ)
	site = @index(Global, Cartesian)

	@unroll for μ in 1:4
        @inbounds U[μ,site] = cmatmul_oo(exp_iQ(-im*ϵ*P[μ,site]), U[μ,site])
    end
end

gaussian_TA!(p::Temporaryfield{GPUD,T}, ϕ) where {T} =
	@latmap(Sequential(), Val(1), gaussian_TA_kernel!, p, T(sqrt(1 - ϕ^2)), T(ϕ), T)

@kernel function gaussian_TA_kernel!(P, ϕ₁, ϕ₂, T)
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
		@inbounds P[μ,site] = ϕ₁*gaussian_TA_mat(T) + ϕ₂*p[μ,site]
	end
end

calc_kinetic_energy(p::Temporaryfield{GPUD}) =
	@latsum(Sequential(), Val(1), calc_kinetic_energy_kernel!, p)

@kernel function calc_kinetic_energy_kernel!(out, @Const(P), neutral)
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	site = @index(Global, Cartesian)

	k = neutral
	@unroll for μ in 1i32:4i32
        @inbounds pmat = P[μ,site]
        k += real(multr(pmat, pmat))
	end

	out_group = @groupreduce(+, k, neutral)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] = out_group
	end
end

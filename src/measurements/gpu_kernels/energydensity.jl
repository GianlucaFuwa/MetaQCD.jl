energy_density(::Plaquette, U::Gaugefield{B}) where {B<:GPU} =
	@latsum(Sequential(), Val(1), energy_density_plaq_kernel!, U) / U.NV

energy_density(::Clover, U::Gaugefield{B}) where {B<:GPU} =
	@latsum(Sequential(), Val(1), energy_density_clov_kernel!, U) / U.NV

energy_density(::Improved, U::Gaugefield{B}) where {B<:GPU} =
	@latsum(Sequential(), Val(1), energy_density_imp_kernel!, U) / U.NV

@kernel function energy_density_plaq_kernel!(out, @Const(U))
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	site = @index(Global, Cartesian)
	T = float_type(U)

	e = T(0.0)
	@inbounds for μ in 1i32:4i32
		for ν in μ+1i32:4i32
			if μ == ν
				continue
			end
			Cμν = plaquette(U, μ, ν, site)
            Fμν = im * traceless_antihermitian(Cμν)
            e += real(multr(Fμν, Fμν))
		end
	end

	out_group = @groupreduce(+, e, T(0.0))

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] = out_group
	end
end

@kernel function energy_density_clov_kernel!(out, @Const(U))
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	site = @index(Global, Cartesian)

	e = 0.0
	@inbounds for μ in 1i32:4i32
		for ν in μ+1i32:4i32
			if μ == ν
				continue
			end
			Cμν = clover_square(U, μ, ν, site, 1)
            Fμν = im/4 * traceless_antihermitian(Cμν)
            e += real(multr(Fμν, Fμν))
		end
	end

	out_group = @groupreduce(+, e, 0.0)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] = out_group
	end
end

@kernel function energy_density_imp_kernel!(out, @Const(U))
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	site = @index(Global, Cartesian)
	T = float_type(U)

	ec = 0.0
    er = 0.0
	@inbounds for μ in 1i32:4i32
		for ν in μ+1i32:4i32
			if μ == ν
				continue
			end
			Cμν = clover_square(U, μ, ν, site, 1)
            Fμν = im/4 * traceless_antihermitian(Cμν)
            ec += real(multr(Fμν, Fμν))
            Cμν = clover_rect(U, μ, ν, site, 1, 2)
            Fμν = im/8 * traceless_antihermitian(Cμν)
            er += real(multr(Fμν, Fμν))
		end
	end

	out_group = @groupreduce(+, 5/3*ec-1/12*er, 0.0)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] = out_group
	end
end

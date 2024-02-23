top_charge(::Plaquette, U::Gaugefield{GPUD}) =
	@latsum(Sequential(), Val(1), top_charge_plaq_kernel!, U) / 4π^2

top_charge(::Clover, U::Gaugefield{GPUD}) =
	@latsum(Sequential(), Val(1), top_charge_clov_kernel!, U) / 4π^2

top_charge(::Improved, U::Gaugefield{GPUD}) =
	@latsum(Sequential(), Val(1), top_charge_imp_kernel!, U) / 4π^2

@kernel function top_charge_plaq_kernel!(out, @Const(U), neutral)
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	site = @index(Global, Cartesian)

	tc = top_charge_density_plaq(U, site)
	out_group = @groupreduce(+, tc, neutral)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] = out_group
	end
end

@kernel function top_charge_clov_kernel!(out, @Const(U), neutral)
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	site = @index(Global, Cartesian)

	tc = top_charge_density_clover(U, site)
	out_group = @groupreduce(+, tc, neutral)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] = out_group
	end
end

@kernel function top_charge_imp_kernel!(out, @Const(U), neutral)
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	site = @index(Global, Cartesian)
	c₁ = convert(eltype(neutral), 5/3)
    c₂ = convert(eltype(neutral), -2/12)

	tc = top_charge_density_imp(U, site, c₁, c₂)
	out_group = @groupreduce(+, tc, neutral)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] = out_group
	end
end

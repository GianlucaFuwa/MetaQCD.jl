wilsonloop(U::Gaugefield{B}, Lμ, Lν) where {B} =
	@latsum(Sequential(), Val(1), wilsonloop_kernel!, U, Lμ, Lν)

@kernel function wilsonloop_kernel!(out, @Const(U), Lμ, Lν)
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	site = @index(Global, Cartesian)
	T = float_type(U)

	wl = T(0.0)
	@unroll for μ in 1i32:3i32
		for ν in μ+1i32:4i32
			wl += real(tr(wilsonloop(U, μ, ν, site, Lμ, Lν)))
		end
	end

	out_group = @groupreduce(+, wl, T(0.0))

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] = out_group
	end
end

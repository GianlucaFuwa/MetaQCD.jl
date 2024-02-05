wilsonloop(U::Gaugefield{GPUD}, Lμ, Lν) = @latreduce(+, wilsonloop_kernel!, U, Lμ, Lν)

@kernel function wilsonloop_kernel!(out, @Const(U), Lμ, Lν, neutral)
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	site = @index(Global, Cartesian)

	wl = neutral
	@unroll for μ in 1i32:3i32
		for ν in μ+1i32:4i32
			wl += real(tr(wilsonloop(U, μ, ν, site, Lμ, Lν)))
		end
	end

	out_group = @groupreduce(+, wl, neutral)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] = out_group
	end
end

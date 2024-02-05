plaquette_trace_sum(U::Gaugefield{GPUD}) = @latreduce(+, plaquette_trace_sum_kernel!, U)

rect_trace_sum(U::Gaugefield{GPUD}) = @latreduce(+, rect_trace_sum_kernel!, U)

@kernel function plaquette_trace_sum_kernel!(out, @Const(U), neutral)
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	site = @index(Global, Cartesian)

	p = neutral
	@unroll for μ in 1:3
		for ν in μ+1:4
			p += real(tr(plaquette(U, μ, ν, site)))
		end
	end

	out_group = @groupreduce(+, p, neutral)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] = out_group
	end
end

@kernel function rect_trace_sum_kernel!(out, @Const(U), @Const(neutral))
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	site = @index(Global, Cartesian)

	r = neutral
	@unroll for μ in 1i32:3i32
		for ν in μ+1i32:4i32
			r += real(tr(rect_1x2(U, μ, ν, site))) + real(tr(rect_2x1(U, μ, ν, site)))
		end
	end

	out_group = @groupreduce(+, r, neutral)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] = out_group
	end
end

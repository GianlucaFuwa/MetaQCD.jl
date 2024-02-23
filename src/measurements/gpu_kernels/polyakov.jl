function polyakov_traced(U::Gaugefield{GPUD,T}) where {T}
    backend = get_backend(U)
    NX, NY, NZ, NT = size(U)[2:end]
	ndrange = (NX, NY, NZ)
	workgroupsize = (4, 4, 4)
	numblocks = cld(prod(ndrange), prod(workgroupsize))
	out = KA.zeros(backend, Complex{T}, numblocks)

    kernel! = polyakov_traced_kernel!(backend, workgroupsize)
    kernel!(out, U.U, NT, zero(Complex{T}), ndrange=ndrange)
	synchronize(backend)
	return sum(out) / (NX * NY * NZ)
end

@kernel function polyakov_traced_kernel!(out, @Const(U), @Const(NT), @Const(neutral))
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	ix, iy, iz = @index(Global, NTuple)

	polymat = U[4,ix,iy,iz,1]
	@unroll for it in 2:NT
		@inbounds polymat = cmatmul_oo(polymat, U[4,ix,iy,iz,it])
	end
    p = tr(polymat)

	out_group = @groupreduce(+, p, neutral)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] = out_group
	end
end

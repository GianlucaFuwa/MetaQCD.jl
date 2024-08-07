function polyakov_traced(U::Gaugefield{B,T}) where {B<:GPU,T}
    NX, NY, NZ, NT = dims(U)
    ndrange = (NX, NY, NZ)
    workgroupsize = (4, 4, 4)
    numblocks = cld(prod(ndrange), prod(workgroupsize))
    out = KA.zeros(B(), ComplexF64, numblocks)

    kernel! = polyakov_traced_kernel!(B(), workgroupsize)
    kernel!(out, U.U, NT; ndrange=ndrange)
    synchronize(B())
    return sum(out) / (NX * NY * NZ)
end

@kernel function polyakov_traced_kernel!(out, @Const(U), @Const(NT))
    # workgroup index, that we use to pass the reduced value to global "out"
    bi = @index(Group, Linear)
    ix, iy, iz = @index(Global, NTuple)

    polymat = U[4, ix, iy, iz, 1]
    @unroll for it in 2:NT
        @inbounds polymat = cmatmul_oo(polymat, U[4, ix, iy, iz, it])
    end
    p = ComplexF64(tr(polymat))

    out_group = @groupreduce(+, p, 0.0 + 0.0im)

    ti = @index(Local)
    if ti == 1
        @inbounds out[bi] = out_group
    end
end

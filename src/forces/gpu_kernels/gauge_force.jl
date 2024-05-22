function calc_dSdU!(
    dU::Temporaryfield{B,T}, staples::Temporaryfield{B,T}, U::Gaugefield{B,T,A,GA}
) where {B<:GPU,T,A,GA}
    check_dims(dU, U, staples)
    fac = convert(T, -U.β / 6)
    @latmap(Sequential(), Val(1), calc_dSdU_kernel!, dU, staples, U, GA(), fac)
    return nothing
end

@kernel function calc_dSdU_kernel!(dU, staples, @Const(U), GA, fac)
    site = @index(Global, Cartesian)

    @inbounds begin
        @unroll for μ in (1i32):(4i32)
            A = staple(GA, U, μ, site)
            staples[μ, site] = A
            UA = cmatmul_od(U[μ, site], A)
            dU[μ, site] = fac * traceless_antihermitian(UA)
        end
    end
end

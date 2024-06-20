function calc_dSdU!(
    dU::Colorfield{B,T}, staples::Colorfield{B,T}, U::Gaugefield{B,T}
) where {B<:GPU,T}
    check_dims(dU, U, staples)
    fac = convert(T, -U.β / 6)
    gaction = gauge_action(U)()
    @latmap(Sequential(), Val(1), calc_dSdU_kernel!, dU, staples, U, gaction, fac)
    return nothing
end

@kernel function calc_dSdU_kernel!(dU, staples, @Const(U), gaction, fac)
    site = @index(Global, Cartesian)

    @inbounds begin
        @unroll for μ in (1i32):(4i32)
            A = staple(gaction, U, μ, site)
            staples[μ, site] = A
            UA = cmatmul_od(U[μ, site], A)
            dU[μ, site] = fac * traceless_antihermitian(UA)
        end
    end
end

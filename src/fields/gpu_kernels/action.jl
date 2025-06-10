function plaquette_trace_sum(U::Gaugefield{B}) where {B}
    return @latsum(
        Sequential(), Val(1), Float64, plaquette_trace_sum_gpu!, U, eachindex(U)
    )
end

@kernel cpu=false function plaquette_trace_sum_gpu!(out, @Const(U), bulk)
    # workgroup index, that we use to pass the reduced value to global "out"
    iblock = @index(Group, Linear)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    pₙ = 0.0
    @unroll for μ in (1i32):(3i32)
        for ν in (μ+1i32):(4i32)
            pₙ += real(tr(plaquette(U, μ, ν, site)))
        end
    end

    out_group = @groupreduce(+, pₙ, 0.0)

    ti = @index(Local)
    if ti == 1
        @inbounds out[iblock] = out_group
    end
end

function rect_trace_sum(U::Gaugefield{B}) where {B}
    return @latsum(
        Sequential(), Val(1), Float64, rect_trace_sum_gpu!, U, eachindex(U)
    )
end

@kernel cpu=false function rect_trace_sum_gpu!(out, @Const(U), bulk)
    # workgroup index, that we use to pass the reduced value to global "out"
    iblock = @index(Group, Linear)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    r = 0.0
    @unroll for μ in (1i32):(3i32)
        for ν in (μ+1i32):(4i32)
            r += real(tr(rect_1x2(U, μ, ν, site))) + real(tr(rect_2x1(U, μ, ν, site)))
        end
    end

    out_group = @groupreduce(+, r, 0.0)

    ti = @index(Local)
    if ti == 1
        @inbounds out[iblock] = out_group
    end
end

function gauge_action_deriv!(
    dU::Colorfield{B,T}, staples::Colorfield{B,T}, U::Gaugefield{B,T}
) where {B<:GPU,T}
    fac = convert(T, -U.β / 6)
    gaction = gauge_action(U)()
    bulk = eachindex(dU, U, staples)
    @latmap(Sequential(), Val(1), gauge_action_deriv_gpu!, dU, staples, U, gaction, fac, bulk)
    return nothing
end

@kernel cpu=false function gauge_action_deriv_gpu!(dU, staples, @Const(U), gaction, fac, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @inbounds begin
        @unroll for μ in (1i32):(4i32)
            A = staple(gaction, U, μ, site)
            staples[μ, site] = A
            UA = cmatmul_od(U[μ, site], A)
            dU[μ, site] = fac * traceless_antihermitian(UA)
        end
    end
end

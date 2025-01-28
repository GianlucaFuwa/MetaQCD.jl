function plaquette_trace_sum(U::Gaugefield{B}) where {B}
    return @latsum(
        Sequential(), Val(1), Float64, plaquette_trace_sum_kernel!, U, eachindex(U)
    )
end

@kernel function plaquette_trace_sum_kernel!(out, @Const(U), bulk_sites)
    # workgroup index, that we use to pass the reduced value to global "out"
    bi = @index(Group, Linear)
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    pₙ = 0.0
    @unroll for μ in (1i32):(3i32)
        for ν in (μ+1i32):(4i32)
            pₙ += real(tr(plaquette(U, μ, ν, site)))
        end
    end

    out_group = @groupreduce(+, pₙ, 0.0)

    ti = @index(Local)
    if ti == 1
        @inbounds out[bi] = out_group
    end
end

function rect_trace_sum(U::Gaugefield{B}) where {B}
    return @latsum(
        Sequential(), Val(1), Float64, rect_trace_sum_kernel!, U, eachindex(U)
    )
end

@kernel function rect_trace_sum_kernel!(out, @Const(U), bulk_sites)
    # workgroup index, that we use to pass the reduced value to global "out"
    bi = @index(Group, Linear)
    site_raw = @index(Global, Cartesian)
    site = bulk_sites[site_raw]

    r = 0.0
    @unroll for μ in (1i32):(3i32)
        for ν in (μ+1i32):(4i32)
            r += real(tr(rect_1x2(U, μ, ν, site))) + real(tr(rect_2x1(U, μ, ν, site)))
        end
    end

    out_group = @groupreduce(+, r, 0.0)

    ti = @index(Local)
    if ti == 1
        @inbounds out[bi] = out_group
    end
end

function gauge_action_deriv!(
    dU::Colorfield{B,T}, staples::Colorfield{B,T}, U::Gaugefield{B,T}
) where {B<:GPU,T}
    check_dims(dU, U, staples)
    fac = convert(T, -U.β / 6)
    gaction = gauge_action(U)()
    @latmap(Sequential(), Val(1), gauge_action_deriv_kernel!, dU, staples, U, gaction, fac)
    return nothing
end

@kernel function gauge_action_deriv_kernel!(dU, staples, @Const(U), gaction, fac)
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

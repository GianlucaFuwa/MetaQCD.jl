function top_charge(::Plaquette, U::Gaugefield{B,T}) where {B<:GPU,T}
    bulk = eachindex(U)
    return @latsum(bulk, Float64, top_charge_plaq_kernel!, U) / 4π^2
end

function top_charge(::Clover, U::Gaugefield{B,T}) where {B<:GPU,T}
    bulk = eachindex(U)
    return @latsum(bulk, Float64, top_charge_clov_kernel!, U, T) / 4π^2
end

function top_charge(::Improved, U::Gaugefield{B,T}) where {B<:GPU,T}
    bulk = eachindex(U)
    return @latsum(bulk, Float64, top_charge_imp_kernel!, U, T) / 4π^2
end

@kernel function top_charge_plaq_kernel!(out, @Const(U), bulk)
    # workgroup index, that we use to pass the reduced value to global "out"
    iblock = @index(Group, Linear)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    tc = top_charge_density_plaq(U, site)
    out_group = @groupreduce(+, tc, 0.0)

    ithread = @index(Local)
    if ithread == 1
        @inbounds out[iblock] = out_group
    end
end

@kernel function top_charge_clov_kernel!(out, @Const(U), ::Type{T}, bulk) where {T}
    # workgroup index, that we use to pass the reduced value to global "out"
    iblock = @index(Group, Linear)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    tc = top_charge_density_clover(U, site, T)
    out_group = @groupreduce(+, tc, 0.0)

    ithread = @index(Local)
    if ithread == 1
        @inbounds out[iblock] = out_group
    end
end

@kernel function top_charge_imp_kernel!(out, @Const(U), ::Type{T}, bulk) where {T}
    # workgroup index, that we use to pass the reduced value to global "out"
    iblock = @index(Group, Linear)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    c₀ = T(5 / 3)
    c₁ = T(-2 / 12)

    tc = top_charge_density_imp(U, site, c₀, c₁, T)
    out_group = @groupreduce(+, tc, 0.0)

    ithread = @index(Local)
    if ithread == 1
        @inbounds out[iblock] = out_group
    end
end

function top_charge_deriv!(
    kind_of_charge, dU::Colorfield{B,T}, F::Tensorfield{B,T}, U::Gaugefield{B,T}, fac=1.0,
) where {B<:GPU,T}
    fac = convert(T, fac / 4π^2)
    bulk = eachindex(dU, U, F)
    fieldstrength_eachsite!(kind_of_charge, F, U)
    @latmap(bulk, top_charge_deriv_kernel!, dU, F, U, kind_of_charge, fac)
    return nothing
end

@kernel function top_charge_deriv_kernel!(dU, @Const(F), @Const(U), kind_of_charge, fac, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @inbounds begin
        tmp1 = cmatmul_oo(
            U[1i32, site],
            (
                ∇trFμνFρσ(kind_of_charge, U, F, 1, 2, 3, 4, site) -
                ∇trFμνFρσ(kind_of_charge, U, F, 1, 3, 2, 4, site) +
                ∇trFμνFρσ(kind_of_charge, U, F, 1, 4, 2, 3, site)
            ),
        )
        dU[1i32, site] = fac * traceless_antihermitian(tmp1)

        tmp2 = cmatmul_oo(
            U[2i32, site],
            (
                ∇trFμνFρσ(kind_of_charge, U, F, 2, 3, 1, 4, site) -
                ∇trFμνFρσ(kind_of_charge, U, F, 2, 1, 3, 4, site) -
                ∇trFμνFρσ(kind_of_charge, U, F, 2, 4, 1, 3, site)
            ),
        )
        dU[2i32, site] = fac * traceless_antihermitian(tmp2)

        tmp3 = cmatmul_oo(
            U[3i32, site],
            (
                ∇trFμνFρσ(kind_of_charge, U, F, 3, 1, 2, 4, site) -
                ∇trFμνFρσ(kind_of_charge, U, F, 3, 2, 1, 4, site) +
                ∇trFμνFρσ(kind_of_charge, U, F, 3, 4, 1, 2, site)
            ),
        )
        dU[3i32, site] = fac * traceless_antihermitian(tmp3)

        tmp4 = cmatmul_oo(
            U[4i32, site],
            (
                ∇trFμνFρσ(kind_of_charge, U, F, 4, 2, 1, 3, site) -
                ∇trFμνFρσ(kind_of_charge, U, F, 4, 1, 2, 3, site) -
                ∇trFμνFρσ(kind_of_charge, U, F, 4, 3, 1, 2, site)
            ),
        )
        dU[4i32, site] = fac * traceless_antihermitian(tmp4)
    end
end

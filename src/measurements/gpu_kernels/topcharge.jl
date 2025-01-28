function top_charge(::Plaquette, U::Gaugefield{B,T}) where {B<:GPU,T}
    return @latsum(Sequential(), Val(1), Float64, top_charge_plaq_kernel!, U) / 4π^2
end

function top_charge(::Clover, U::Gaugefield{B,T}) where {B<:GPU,T}
    return @latsum(Sequential(), Val(1), Float64, top_charge_clov_kernel!, U, T) / 4π^2
end

function top_charge(::Improved, U::Gaugefield{B,T}) where {B<:GPU,T}
    return @latsum(Sequential(), Val(1), Float64, top_charge_imp_kernel!, U, T) / 4π^2
end

@kernel function top_charge_plaq_kernel!(out, @Const(U))
    # workgroup index, that we use to pass the reduced value to global "out"
    bi = @index(Group, Linear)
    site = @index(Global, Cartesian)

    tc = top_charge_density_plaq(U, site)
    out_group = @groupreduce(+, tc, 0.0)

    ti = @index(Local)
    if ti == 1
        @inbounds out[bi] = out_group
    end
end

@kernel function top_charge_clov_kernel!(out, @Const(U), ::Type{T}) where {T}
    # workgroup index, that we use to pass the reduced value to global "out"
    bi = @index(Group, Linear)
    site = @index(Global, Cartesian)

    tc = top_charge_density_clover(U, site, T)
    out_group = @groupreduce(+, tc, 0.0)

    ti = @index(Local)
    if ti == 1
        @inbounds out[bi] = out_group
    end
end

@kernel function top_charge_imp_kernel!(out, @Const(U), ::Type{T}) where {T}
    # workgroup index, that we use to pass the reduced value to global "out"
    bi = @index(Group, Linear)
    site = @index(Global, Cartesian)
    c₀ = T(5 / 3)
    c₁ = T(-2 / 12)

    tc = top_charge_density_imp(U, site, c₀, c₁, T)
    out_group = @groupreduce(+, tc, 0.0)

    ti = @index(Local)
    if ti == 1
        @inbounds out[bi] = out_group
    end
end

function top_charge_deriv!(
    kind_of_charge, dU::Colorfield{B,T}, F::Tensorfield{B,T}, U::Gaugefield{B,T}, fac=1.0,
) where {B<:GPU,T}
    check_dims(dU, U, F)
    fac = convert(T, fac / 4π^2)
    fieldstrength_eachsite!(kind_of_charge, F, U)
    @latmap(Sequential(), Val(1), top_charge_deriv_kernel!, dU, F, U, kind_of_charge, fac)
    return nothing
end

@kernel function top_charge_deriv_kernel!(dU, @Const(F), @Const(U), kind_of_charge, fac)
    site = @index(Global, Cartesian)

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

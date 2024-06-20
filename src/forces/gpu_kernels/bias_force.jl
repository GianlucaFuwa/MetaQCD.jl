function calc_dQdU!(
    kind_of_charge, dU::Colorfield{B,T}, F::Tensorfield{B,T}, U::Gaugefield{B,T}, fac=1.0,
) where {B<:GPU,T}
    check_dims(dU, U, F)
    fac = convert(T, fac / 4π^2)
    @latmap(Sequential(), Val(1), calc_dQdU_kernel!, dU, F, U, kind_of_charge, fac)
    return nothing
end

@kernel function calc_dQdU_kernel!(dU, @Const(F), @Const(U), kind_of_charge, fac)
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

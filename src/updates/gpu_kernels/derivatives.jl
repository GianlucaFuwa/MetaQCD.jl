function calc_dSdU!(dU::Temporaryfield{GPUD,T}, staples::Temporaryfield{GPUD,T},
    U::Gaugefield{GPUD,T,A,GA}) where {T,A,GA}
    @assert size(U) == size(dU) == size(staples)
    @latmap(Sequential(), Val(1), calc_dSdU_kernel!, dU, staples, U, GA(), T(U.β))
    return nothing
end

@kernel function calc_dSdU_kernel!(dU, staples, @Const(U), @Const(GA), @Const(β))
	site = @index(Global, Cartesian)

	@unroll for μ in 1:4
        A = staple(GA, U, μ, site)
        @inbounds staples[μ,site] = A
        UA = @inbounds cmatmul_od(U[μ,site], A)
        @inbounds dU[μ,site] = -β/6 * traceless_antihermitian(UA)
    end
end

function calc_dQdU!(kind_of_charge, dU::Temporaryfield{GPUD,T}, F::Tensorfield{GPUD,T},
    U::Gaugefield{GPUD,T}) where {T}
    @assert size(U) == size(dU) == size(F)
    @latmap(Sequential(), Val(1), calc_dQdU_kernel!, dU, F, U, kind_of_charge, T(1.0))
    return nothing
end

@kernel function calc_dQdU_kernel!(dU, @Const(F), @Const(U), kind_of_charge, fac)
	site = @index(Global, Cartesian)
    c = fac / 4π^2

	@inbounds begin
        tmp1 = cmatmul_oo(U[1,site], (∇trFμνFρσ(kind_of_charge, U, F, 1, 2, 3, 4, site) -
                                      ∇trFμνFρσ(kind_of_charge, U, F, 1, 3, 2, 4, site) +
                                      ∇trFμνFρσ(kind_of_charge, U, F, 1, 4, 2, 3, site)))
        dU[1,site] = c * traceless_antihermitian(tmp1)

        tmp2 = cmatmul_oo(U[2,site], (∇trFμνFρσ(kind_of_charge, U, F, 2, 3, 1, 4, site) -
                                      ∇trFμνFρσ(kind_of_charge, U, F, 2, 1, 3, 4, site) -
                                      ∇trFμνFρσ(kind_of_charge, U, F, 2, 4, 1, 3, site)))
        dU[2,site] = c * traceless_antihermitian(tmp2)

        tmp3 = cmatmul_oo(U[3,site], (∇trFμνFρσ(kind_of_charge, U, F, 3, 1, 2, 4, site) -
                                      ∇trFμνFρσ(kind_of_charge, U, F, 3, 2, 1, 4, site) +
                                      ∇trFμνFρσ(kind_of_charge, U, F, 3, 4, 1, 2, site)))
        dU[3,site] = c * traceless_antihermitian(tmp3)

        tmp4 = cmatmul_oo(U[4,site], (∇trFμνFρσ(kind_of_charge, U, F, 4, 2, 1, 3, site) -
                                      ∇trFμνFρσ(kind_of_charge, U, F, 4, 1, 2, 3, site) -
                                      ∇trFμνFρσ(kind_of_charge, U, F, 4, 3, 1, 2, site)))
        dU[4,site] = c * traceless_antihermitian(tmp4)
    end
end

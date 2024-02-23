function fieldstrength_eachsite!(::Plaquette, F::Tensorfield{GPUD}, U::Gaugefield{GPUD})
    @assert size(F) == size(U)
    @latmap(Sequential(), Val(1), fieldstrength_eachsite_Pkernel!, F, U)
    return nothing
end

@kernel function fieldstrength_eachsite_Pkernel!(F, @Const(U))
	site = @index(Global, Cartesian)

	@inbounds begin
        C12 = plaquette(U, 1, 2, site)
        F[1,2,site] = im * traceless_antihermitian(C12)
        C13 = plaquette(U, 1, 3, site)
        F[1,3,site] = im * traceless_antihermitian(C13)
        C14 = plaquette(U, 1, 4, site)
        F[1,4,site] = im * traceless_antihermitian(C14)
        C23 = plaquette(U, 2, 3, site)
        F[2,3,site] = im * traceless_antihermitian(C23)
        C24 = plaquette(U, 2, 4, site)
        F[2,4,site] = im * traceless_antihermitian(C24)
        C34 = plaquette(U, 3, 4, site)
        F[3,4,site] = im * traceless_antihermitian(C34)
    end
end

function fieldstrength_eachsite!(::Clover, F::Tensorfield{GPUD}, U::Gaugefield{GPUD})
    @assert size(F) == size(U)
    @latmap(Sequential(), Val(1), fieldstrength_eachsite_Ckernel!, F, U)
    return nothing
end

@kernel function fieldstrength_eachsite_Ckernel!(F, @Const(U))
	site = @index(Global, Cartesian)

	@inbounds begin
        C12 = clover_square(U, 1, 2, site, 1)
        F[1,2,site] = im/4 * traceless_antihermitian(C12)
        C13 = clover_square(U, 1, 3, site, 1)
        F[1,3,site] = im/4 * traceless_antihermitian(C13)
        C14 = clover_square(U, 1, 4, site, 1)
        F[1,4,site] = im/4 * traceless_antihermitian(C14)
        C23 = clover_square(U, 2, 3, site, 1)
        F[2,3,site] = im/4 * traceless_antihermitian(C23)
        C24 = clover_square(U, 2, 4, site, 1)
        F[2,4,site] = im/4 * traceless_antihermitian(C24)
        C34 = clover_square(U, 3, 4, site, 1)
        F[3,4,site] = im/4 * traceless_antihermitian(C34)
    end
end

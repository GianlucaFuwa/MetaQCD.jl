function fieldstrength_eachsite!(::Plaquette, F::Tensorfield{B}, U::Gaugefield{B}) where {B}
    @assert dims(F) == dims(U)
    @latmap(Sequential(), Val(1), fieldstrength_eachsite_Pkernel!, F, U)
    return nothing
end

@kernel function fieldstrength_eachsite_Pkernel!(F, @Const(U))
	site = @index(Global, Cartesian)

	@inbounds begin
        C12 = plaquette(U, 1i32, 2i32, site)
        F[1i32,2i32,site] = im * traceless_antihermitian(C12)
        C13 = plaquette(U, 1i32, 3i32, site)
        F[1i32,3i32,site] = im * traceless_antihermitian(C13)
        C14 = plaquette(U, 1i32, 4i32, site)
        F[1i32,4i32,site] = im * traceless_antihermitian(C14)
        C23 = plaquette(U, 2i32, 3i32, site)
        F[2i32,3i32,site] = im * traceless_antihermitian(C23)
        C24 = plaquette(U, 2i32, 4i32, site)
        F[2i32,4i32,site] = im * traceless_antihermitian(C24)
        C34 = plaquette(U, 3i32, 4i32, site)
        F[3i32,4i32,site] = im * traceless_antihermitian(C34)
    end
end

function fieldstrength_eachsite!(::Clover, F::Tensorfield{B}, U::Gaugefield{B}) where {B}
    @assert dims(F) == dims(U)
    @latmap(Sequential(), Val(1), fieldstrength_eachsite_Ckernel!, F, U)
    return nothing
end

@kernel function fieldstrength_eachsite_Ckernel!(F, @Const(U))
	site = @index(Global, Cartesian)
    onefourth = float_type(U)(1/4)

	@inbounds begin
        C12 = clover_square(U, 1i32, 2i32, site, 1i32)
        F[1i32,2i32,site] = im * onefourth * traceless_antihermitian(C12)
        C13 = clover_square(U, 1i32, 3i32, site, 1i32)
        F[1i32,3i32,site] = im * onefourth * traceless_antihermitian(C13)
        C14 = clover_square(U, 1i32, 4i32, site, 1i32)
        F[1i32,4i32,site] = im * onefourth * traceless_antihermitian(C14)
        C23 = clover_square(U, 2i32, 3i32, site, 1i32)
        F[2i32,3i32,site] = im * onefourth * traceless_antihermitian(C23)
        C24 = clover_square(U, 2i32, 4i32, site, 1i32)
        F[2i32,4i32,site] = im * onefourth * traceless_antihermitian(C24)
        C34 = clover_square(U, 3i32, 4i32, site, 1i32)
        F[3i32,4i32,site] = im * onefourth * traceless_antihermitian(C34)
    end
end

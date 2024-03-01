function update!(parity::ParityUpdate, U::Gaugefield{B}) where {B<:GPU}
    @assert typeof(U) == typeof(parity.U_bak)
    U_bak = parity.U_bak
    @assert dims(U_bak) == dims(U)
    @latmap(Sequential(), Val(1), parity_update_kernel!, U, U_bak)
    return nothing
end

@kernel function parity_update_kernel!(U, @Const(U_bak))
	ix, iy, iz, it = @index(Global, NTuple)

	@inbounds begin
        ix_min_0 = mod(-ix, NX) + 1i32
        ix_min_1 = mod(-ix-1i32, NX) + 1i32
        iy_min_0 = mod(-iy, NY) + 1i32
        iy_min_1 = mod(-iy-1i32, NY) + 1i32
        iz_min_0 = mod(-iz, NZ) + 1i32
        iz_min_1 = mod(-iz-1i32, NZ) + 1i32
        U[1,ix,iy,iz,it] = U_bak[1,ix_min_1,iy_min_0,iz_min_0,it]'
        U[2,ix,iy,iz,it] = U_bak[2,ix_min_0,iy_min_1,iz_min_0,it]'
        U[3,ix,iy,iz,it] = U_bak[3,ix_min_0,iy_min_0,iz_min_1,it]'
        U[4,ix,iy,iz,it] = U_bak[4,ix_min_0,iy_min_0,iz_min_0,it]
	end
end

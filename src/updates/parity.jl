struct ParityUpdate{TG} <: AbstractUpdate
    U_bak::TG
    function ParityUpdate(U::TG) where {TG}
        @level1("[ Parity update enabled\n")
        return new{TG}(similar(U))
    end
end

function update!(parity::ParityUpdate, U::Gaugefield{B,T,M}) where {B,T,M}
    NX, NY, NZ, _ = size(U)
    U_bak = parity.U_bak
    update_halo!(U)
    copy!(U_bak, U)

    parallelfor(eachindex(U), B, Val(M), (U,), (U,), (U,)) do site, (U,)
        ix, iy, iz, it = site.I
        ix_min_0 = mod(-ix, NX) + 1
        ix_min_1 = mod(-ix - 1, NX) + 1
        iy_min_0 = mod(-iy, NY) + 1
        iy_min_1 = mod(-iy - 1, NY) + 1
        iz_min_0 = mod(-iz, NZ) + 1
        iz_min_1 = mod(-iz - 1, NZ) + 1
        U[1, site] = U_bak[1, CartesianIndex(ix_min_1, iy_min_0, iz_min_0, it)]'
        U[2, site] = U_bak[2, CartesianIndex(ix_min_0, iy_min_1, iz_min_0, it)]'
        U[3, site] = U_bak[3, CartesianIndex(ix_min_0, iy_min_0, iz_min_1, it)]'
        U[4, site] = U_bak[4, CartesianIndex(ix_min_0, iy_min_0, iz_min_0, it)]
    end

    return nothing
end

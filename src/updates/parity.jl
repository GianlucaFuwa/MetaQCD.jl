struct ParityUpdate{TG} <: AbstractUpdate
    _temp_U::TG

    ParityUpdate(U::TG) where {TG} = new{TG}(similar(U))
end

function update!(updatemethod::ParityUpdate, U)
    NX, NY, NZ, NT = size(U)
    U_bak = updatemethod._temp_U
    substitute_U!(U_bak, U)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    ix_min_0 = mod(-ix-0, NX) + 1
                    ix_min_1 = mod(-ix-1, NX) + 1
                    iy_min_0 = mod(-iy-0, NY) + 1
                    iy_min_1 = mod(-iy-1, NY) + 1
                    iz_min_0 = mod(-iz-0, NZ) + 1
                    iz_min_1 = mod(-iz-1, NZ) + 1
                    U[1][ix,iy,iz,it] = U_bak[1][ix_min_1,iy_min_0,iz_min_0,it]'
                    U[2][ix,iy,iz,it] = U_bak[2][ix_min_0,iy_min_1,iz_min_0,it]'
                    U[3][ix,iy,iz,it] = U_bak[3][ix_min_0,iy_min_0,iz_min_1,it]'
                    U[4][ix,iy,iz,it] = U_bak[4][ix_min_0,iy_min_0,iz_min_0,it]
                end
            end
        end
    end

    U.CV *= -1
    return nothing
end

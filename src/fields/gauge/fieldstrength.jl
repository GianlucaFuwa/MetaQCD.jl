abstract type AbstractFieldstrength end

struct PlaquetteFieldstrength <: AbstractFieldstrength end
struct CloverFieldstrength <: AbstractFieldstrength end
struct ImprovedFieldstrength <: AbstractFieldstrength end

function fieldstrength_eachsite!(F::Vector{Temporaryfield}, U, name)
    if name == "plaquette"
        fieldstrength_eachsite_plaq!(F, U)
    elseif name == "clover" 
        fieldstrength_eachsite_clover!(F, U)
    #elseif name == "improved" 
    #    fieldstrength_eachsite_improved!(Fμν, u)
    end
    
    return nothing
end

function fieldstrength_eachsite_plaq!(F::Vector{Temporaryfield}, U)
    NX, NY, NZ, NT = size(U)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    C12 = plaquette(U, 1, 2, site)
                    F[1][2][site] = im * traceless_antihermitian(C12)

                    C13 = plaquette(U, 1, 3, site)
                    F[1][3][site] = im * traceless_antihermitian(C13)
                    
                    C14 = plaquette(U, 1, 4, site)
                    F[1][4][site] = im * traceless_antihermitian(C14)
                    
                    C23 = plaquette(U, 2, 3, site)
                    F[2][3][site] = im * traceless_antihermitian(C23)

                    C24 = plaquette(U, 2, 4, site)
                    F[2][4][site] = im * traceless_antihermitian(C24)

                    C34 = plaquette(U, 3, 4, site)
                    F[3][4][site] = im * traceless_antihermitian(C34)
                end
            end
        end
    end

    return nothing
end

function fieldstrength_eachsite_clover!(Fμν::Vector{Temporaryfield}, U)
    NX, NY, NZ, NT = size(U)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    C12 = clover_square(U, 1, 2, site, 1)
                    Fμν[1][2][site] = im/4 * traceless_antihermitian(C12)

                    C13 = clover_square(U, 1, 3, site, 1)
                    Fμν[1][3][site] = im/4 * traceless_antihermitian(C13)

                    C14 = clover_square(U, 1, 4, site, 1)
                    Fμν[1][4][site] = im/4 * traceless_antihermitian(C14)

                    C23 = clover_square(U, 2, 3, site, 1)
                    Fμν[2][3][site] = im/4 * traceless_antihermitian(C23)

                    C24 = clover_square(U, 2, 4, site, 1)
                    Fμν[2][4][site] = im/4 * traceless_antihermitian(C24)

                    C34 = clover_square(U, 3, 4, site, 1)
                    Fμν[3][4][site] = im/4 * traceless_antihermitian(C34)
                end
            end
        end
    end

    return nothing
end
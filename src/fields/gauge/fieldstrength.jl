function fieldstrength_eachsite!(Fμν::Vector{TemporaryField}, u::Gaugefield, name::String)
    if name == "plaquette"
        fieldstrength_eachsite_plaq!(Fμν, u)
    elseif name == "clover" 
        fieldstrength_eachsite_clover!(Fμν, u)
    #elseif name == "improved" 
    #    fieldstrength_eachsite_improved!(Fμν, u)
    else 
        error("Fieldstrength tensor type $(name) is not supported!")
    end
    
    return nothing
end

function fieldstrength_eachsite_plaq!(Fμν::Vector{TemporaryField}, u::Gaugefield)
    NX, NY, NZ, NT = size(u)

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    C12 = plaquette(u, 1, 2, site)
                    Fμν[1][2][ix,iy,iz,it] = im/4 * traceless_antihermitian(C12)

                    C13 = plaquette(u, 1, 3, site)
                    Fμν[1][3][ix,iy,iz,it] = im/4 * traceless_antihermitian(C13)

                    C14 = plaquette(u, 1, 4, site)
                    Fμν[1][4][ix,iy,iz,it] = im/4 * traceless_antihermitian(C14)

                    C23 = plaquette(u, 2, 3, site)
                    Fμν[2][3][ix,iy,iz,it] = im/4 * traceless_antihermitian(C23)

                    C24 = plaquette(u, 2, 4, site)
                    Fμν[2][4][ix,iy,iz,it] = im/4 * traceless_antihermitian(C24)

                    C34 = plaquette(u, 3, 4, site)
                    Fμν[3][4][ix,iy,iz,it] = im/4 * traceless_antihermitian(C34)
                end
            end
        end
    end

    return nothing
end

function fieldstrength_eachsite_clover!(Fμν::Vector{TemporaryField}, u::Gaugefield)
    NX, NY, NZ, NT = size(u)

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    C12 = clover_square(u, 1, 2, site, 1)
                    Fμν[1][2][ix,iy,iz,it] = im/4 * traceless_antihermitian(C12)

                    C13 = clover_square(u, 1, 3, site, 1)
                    Fμν[1][3][ix,iy,iz,it] = im/4 * traceless_antihermitian(C13)

                    C14 = clover_square(u, 1, 4, site, 1)
                    Fμν[1][4][ix,iy,iz,it] = im/4 * traceless_antihermitian(C14)

                    C23 = clover_square(u, 2, 3, site, 1)
                    Fμν[2][3][ix,iy,iz,it] = im/4 * traceless_antihermitian(C23)

                    C24 = clover_square(u, 2, 4, site, 1)
                    Fμν[2][4][ix,iy,iz,it] = im/4 * traceless_antihermitian(C24)

                    C34 = clover_square(u, 3, 4, site, 1)
                    Fμν[3][4][ix,iy,iz,it] = im/4 * traceless_antihermitian(C34)
                end
            end
        end
    end

    return nothing
end
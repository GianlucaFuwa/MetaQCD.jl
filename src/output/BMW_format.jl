function saveU_bmw(U, filename)
    NX,NY,NZ,NT = size(U)

    fp = open(filename, "w")
    i = 0
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for μ = 1:4
                        for a = 1:3
                            for b = 1:3
                                # TODO
                            end
                        end
                    end
                end
            end
        end
    end
    close(fp)
    return nothing
end

function loadU_bmw!(U, initial)
    NX,NY,NZ,NT = size(U)
    fp = open(initial, "r")
    numdata = countlines(initial)
    @assert numdata == 4 * U.NV * 9 * 2 "data shape is wrong"
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for μ = 1:4
                        link = zeros(ComplexF64,3,3)
                        for a = 1:3
                            for b = 1:3
                                # TODO
                            end
                        end
                        U[μ][ix,iy,iz,it] = link
                    end
                end
            end
        end
    end
    close(fp)
end

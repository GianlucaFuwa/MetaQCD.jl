using DelimitedFiles
function load_BridgeText!(initial, U)
    NX,NY,NZ,NT = size(U)
    fp = readdlm(initial)
    numdata = length(fp)
    @assert numdata == 4 * U.NV * 9 * 2 "data shape is wrong"
    i = 0
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    link = zeros(ComplexF64,3,3)
                    for μ = 1:4
                        #
                        for a = 1:3
                            for b = 1:3
                                i += 1
                                u = fp[i]
                                rvalue = u
                                i += 1
                                u = fp[i]
                                ivalue = u
                                link[a,b] = rvalue + im*ivalue
                            end
                        end
                        #
                        if μ==2
                        U[μ][ix,iy,iz,it] = link
                        end
                    end
                end
            end
        end
    end
    return i
end
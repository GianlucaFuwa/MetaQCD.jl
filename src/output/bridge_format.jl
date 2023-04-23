module Bridge_format
    using LinearAlgebra
    export save_textdata
    export load_BridgeText!
    function save_textdata(U, filename)
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
                                    i += 1
                                    rvalue = real(U[μ][ix,iy,iz,it][a,b])
                                    println(fp, rvalue)
                                    ivalue = imag(U[μ][ix,iy,iz,it][a,b])
                                    println(fp, ivalue)
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

    function load_BridgeText!(initial, U)
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
                                    u = readline(fp)
                                    rvalue = parse(Float64, u)
                                    u = readline(fp)
                                    ivalue = parse(Float64, u)
                                    link[a,b] = rvalue + im*ivalue
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

end

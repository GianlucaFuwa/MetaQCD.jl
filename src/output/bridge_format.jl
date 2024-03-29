function saveU(::BridgeFormat, U, filename)
    NX, NY, NZ, NT = size(U)
    fp = open(filename, "w")

    i = 0
    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        for a in 1:3
                            for b in 1:3
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

function loadU!(::BridgeFormat, U, filename)
    NX, NY, NZ, NT = size(U)
    fp = open(filename, "r")
    numdata = countlines(filename)
    @assert numdata == 4*U.NV*9*2 "data shape is wrong"

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        link = zeros(Complex, 3, 3)
                        for a in 1:3
                            for b in 1:3
                                u = readline(fp)
                                rvalue = parse(Float64, u)
                                u = readline(fp)
                                ivalue = parse(Float64, u)
                                link[a, b] = rvalue + im*ivalue
                            end
                        end
                        U[μ][ix,iy,iz,it] = SMatrix{3,3,ComplexF64,9}(link)
                    end
                end
            end
        end
    end

    close(fp)
    return nothing
end

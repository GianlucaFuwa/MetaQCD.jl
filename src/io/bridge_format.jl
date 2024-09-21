function save_config(
    ::BridgeFormat, U::Gaugefield{B,T,false}, filename, args...
) where {B,T}
    @assert U.U isa Array
    NX, NY, NZ, NT = (U.NX, U.NY, U.NZ, U.NT)
    fp = open(filename, "w")

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        for a in 1:3
                            for b in 1:3
                                rvalue = real(U[μ, ix, iy, iz, it][a, b])
                                println(fp, rvalue)
                                ivalue = imag(U[μ, ix, iy, iz, it][a, b])
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

function load_config!(::BridgeFormat, U::Gaugefield{B,T,false}, filename) where {B,T}
    @assert U.U isa Array
    NX, NY, NZ, NT = (U.NX, U.NY, U.NZ, U.NT)
    fp = open(filename, "r")
    numdata = countlines(filename)
    @assert numdata == 4 * U.NV * 9 * 2 "data shape is wrong"

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        link = @MMatrix zeros(Complex{T}, 3, 3)
                        for a in 1:3
                            for b in 1:3
                                u = readline(fp)
                                rvalue = parse(T, u)
                                u = readline(fp)
                                ivalue = parse(T, u)
                                link[a, b] = rvalue + im * ivalue
                            end
                        end
                        U[μ, ix, iy, iz, it] = SMatrix{3,3,ComplexF64,9}(link)
                    end
                end
            end
        end
    end

    close(fp)
    return nothing
end

function save_field(
    ::BridgeFormat, U::AbstractField{B,T,false}, filename, args...
) where {B,T}
    @assert B == CPU
    fp = open(filename, "w")
    Utmp = to_backend(CPU, U)

    for site in eachindex(Utmp)
        for μ in 1:4
            for a in 1:3
                for b in 1:3
                    rvalue = real(Utmp[μ, site][a, b])
                    println(fp, rvalue)
                    ivalue = imag(Utmp[μ, site][a, b])
                    println(fp, ivalue)
                end
            end
        end
    end

    close(fp)
    return nothing
end

function save_field(
    ::BridgeFormat, f::Spinorfield{B,T,false,ND}, filename, args...
) where {B,T,ND}
    @assert B == CPU
    fp = open(filename, "w")
    ftmp = to_backend(CPU, f)

    for site in eachindex(ftmp)
        for a in 1:3ND
            rvalue = real(ftmp[site][a])
            println(fp, rvalue)
            ivalue = imag(ftmp[site][a])
            println(fp, ivalue)
        end
    end

    close(fp)
    return nothing
end

function load_field!(::BridgeFormat, U::Gaugefield{B,T,false}, filename) where {B,T}
    @assert B == CPU "load_field! in bridge format not supported for GPU fields yet"
    fp = open(filename, "r")
    numdata = countlines(filename)
    @assert numdata == 4 * length(U) * 9 * 2 "data shape is wrong"

    for site in eachindex(U)
        for μ in 1:4
            link = zero(MMatrix{3,3,Complex{T},9})
            for a in 1:3
                for b in 1:3
                    u = readline(fp)
                    rvalue = parse(T, u)
                    u = readline(fp)
                    ivalue = parse(T, u)
                    link[a, b] = rvalue + im * ivalue
                end
            end
            U[μ, site] = SMatrix{3,3,ComplexF64,9}(link)
        end
    end

    close(fp)
    return nothing
end

function load_field!(
    ::BridgeFormat, f::Spinorfield{B,T,false,ND}, filename
) where {B,T,ND}
    @assert B == CPU "load_field! in bridge format not supported for GPU fields yet"
    fp = open(filename, "r")
    numdata = countlines(filename)
    @assert numdata == length(f) * 3ND * 2 "data shape is wrong"

    for site in eachindex(f)
        link = zero(MVector{3ND,Complex{T}})
        for a in 1:3ND
            u = readline(fp)
            rvalue = parse(T, u)
            u = readline(fp)
            ivalue = parse(T, u)
            link[a] = rvalue + im * ivalue
        end
        f[site] = SVector{3ND,ComplexF64}(link)
    end

    close(fp)
    return nothing
end

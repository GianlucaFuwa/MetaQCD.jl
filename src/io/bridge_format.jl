function save_field(
    ::BridgeFormat, U::AbstractField{B,T,false}, filename, args...
) where {B,T}
    filename != "" || return nothing
    fp = open(filename, "w")
    Utmp = if B == CPU
        U
    else
        convert_field(CPU, U)
    end

    for site in eachindex(Utmp)
        for μ in 1:4
            Un = Utmp[μ, site]
            for a in 1:3
                for b in 1:3
                    rvalue = Float64(real(Un[a, b]))
                    println(fp, rvalue)
                    ivalue = Float64(imag(Un[a, b]))
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
    filename != "" || return nothing
    fp = open(filename, "w")
    ftmp = if B == CPU
        f
    else
        convert_field(CPU, f)
    end

    for site in eachindex(ftmp)
        fn = ftmp[site]
        for a in 1:3ND
            rvalue = Float64(real(fn[a]))
            println(fp, rvalue)
            ivalue = Float64(imag(fn[a]))
            println(fp, ivalue)
        end
    end

    invalidate_halo!(f) 
    close(fp)
    return nothing
end

function load_field!(::BridgeFormat, U::Gaugefield{CPU,T,false}, filename) where {T}
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
            U[μ, site] = SMatrix{3,3,Complex{T},9}(link)
        end
    end

    invalidate_halo!(U) 
    close(fp)
    return nothing
end

function load_field!(::BridgeFormat, U::Gaugefield{B,T,false,GA,N}, filename) where {B,T,GA,N}
    Ucpu = Gaugefield{CPU,Float64,GA,N}(size(U)..., U.β)
    load_field!(BridgeFormat(), Ucpu, filename)
    Ugpu = convert_field(B, Ucpu, T)
    copy!(U, Ugpu)
    return nothing
end

function load_field!(
    ::BridgeFormat, f::Spinorfield{CPU,T,false,ND}, filename
) where {T,ND}
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
        f[site] = SVector{3ND,Complex{T}}(link)
    end

    close(fp)
    return nothing
end

function load_field!(::BridgeFormat, f::Spinorfield{B,T,false,ND}, filename) where {B,T,ND}
    fcpu = Spinorfield{CPU,Float64,ND}(size(f)...)
    load_field!(BridgeFormat(), fcpu, filename)
    fgpu = convert_field(B, fcpu, T)
    copy!(f, fgpu)
    return nothing
end

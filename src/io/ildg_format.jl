#+---------------------------------------------------------------------------------+
#| This file provides functions to load and save configurations stored in the ILDG |
#| format, see https://hpc.desy.de/ildg/documentation/ for documentation           |
#|---------------------------------------------------------------------------------+

function save_config(::ILDGFormat, U, filename; parameters=nothing, override=false)
    @assert U.U isa Array

    if override == false
        @assert !isfile(filename) """
        File $filename to store config in already exists and override is \"false\"
        """
    end

    fp = open(filename, "w")
    N = U.NC
    @assert N == 3 "Only SU(3) is supported in BMW format"

    ### Header
    header_buf = IOBuffer()
    checksum = Adler64Checksum()

    for site in eachindex(U)
        buffer = MMatrix{12,4,UInt64,48}(undef)
        for μ in 1:4
            tmp = U[μ, site]
            view(buffer, :, μ) .= reinterpret(reshape, UInt64, deconstruct_mat_bmw(tmp))
        end
        site_abs = (((site[4] * U.NZ + site[3]) * U.NY + site[2]) * U.NX + site[1])
        adler64_add!(checksum, buffer, site_abs, U.NV, 16N)
    end

    adler64_finalize!(checksum)
    checksum_str = adler64_string(checksum)
    bytes_written = 0
    bytes_written += write(
        header_buf, "#BMW $(U.NX) $(U.NY) $(U.NZ) $(U.NT) beta_$(U.β) prec_float64 $checksum_str\n"
    )
    bytes_written += write(
        header_buf, "Generated with MetaQCD.jl 1.0.0 on Julia $(VERSION)\n"
    )
    bytes_written += write(header_buf, "@ $(Dates.now())\n")

    while bytes_written < 4095
        bytes_written += write(header_buf, " ")
    end

    write(header_buf, "\n")
    bw = write(fp, String(take!(header_buf)))
    @assert bw == 4096 "Header is not 4096 bytes long"
    ### Links
    for site in eachindex(U)
        for μ in 1:4
            tmp = U[μ, site]
            buf = deconstruct_mat_bmw(tmp)
            for val in buf
                write(fp, hton(val))
            end
        end
    end

    close(fp)
    return nothing
end

function load_config!(::BMWFormat, U, filename)
    @assert U.U isa Array
    fp = open(filename, "r")
    header_bin = Vector{UInt8}(undef, 4096)
    readbytes!(fp, header_bin, 4096)
    header = String(header_bin)
    split_header = split(header)
    @assert split_header[1] == "#BMW" "Header doesn't start with \"#BMW\""
    NX, NY, NZ, NT = parse.(Int, split_header[2:5])
    @assert (NX, NY, NZ, NT) == (U.NX, U.NY, U.NZ, U.NT) "Dimensions do not match"
    N = U.NC
    @assert N == 3 "Only SU(3) is supported in BMW format"
    T = real(eltype(U[1, 1, 1, 1, 1]))
    checksum_read = parse(UInt64, split_header[6])
    checksum_calc = Adler64Checksum()

    for site in eachindex(U)
        buf1 = MMatrix{12,4,Float64,48}(undef)
        buffer = MMatrix{12,4,UInt64,48}(undef)
        read!(fp, buf1)
        for μ in 1:4
            buf2 = MVector{12,Float64}(net_to_host(buf1[:, μ]))
            mat = reconstruct_mat_bmw(buf2, T)
            U[μ, site] = restore_last_row(mat)
            tmp = U[μ, site]
            view(buffer, :, μ) .= reinterpret(reshape, UInt64, deconstruct_mat_bmw(tmp))
        end
        site_abs = (((site[4] * U.NZ + site[3]) * U.NY + site[2]) * U.NX + site[1])
        adler64_add!(checksum_calc, buffer, site_abs, U.NV, 16N)
    end

    close(fp)
    adler64_finalize!(checksum_calc)
    @assert checksum_read == checksum_calc.final """
    Checksums do not match (read: $checksum_read, calculated: $(checksum_calc.final))
    """
    return nothing
end

mutable struct Adler64Checksum
    adler_prime::UInt32
    final::UInt64
    A::UInt64
    B::UInt64
    Adler64Checksum() = new(UInt32(4294967291), 1, 1, 0)
end

function adler64_add!(checksum::Adler64Checksum, data, block_id, num_blocks, block_size)
    sum_a = zero(UInt64)
    sum_b = zero(UInt64)
    revi = (num_blocks - block_id) * 2block_size

    for i in 1:block_size
        d₀ = data[i] >> 32
        d₁ = data[i] & 0xffffffff
        sum_a += d₀ + d₁
        sum_b += (2 + revi * d₀ + (revi - 1) * d₁) % checksum.adler_prime
        revi -= 2
    end

    checksum.A = (checksum.A + sum_a) % checksum.adler_prime
    checksum.B = (checksum.B + sum_b) % checksum.adler_prime
    return nothing
end

function adler64_finalize!(checksum::Adler64Checksum)
    A_copy = (checksum.A + 1) % checksum.adler_prime
    B_copy = checksum.B % checksum.adler_prime
    checksum.final = (B_copy << 32) + A_copy
    return nothing
end

function adler64_string(checksum::Adler64Checksum)
    @assert checksum.final !== checksum.adler_prime "Invalid final in checksum"
    return string(checksum.final)
end

host_to_net(buffer) = ntuple(i -> hton(buffer[i]), length(buffer))
net_to_host(buffer) = ntuple(i -> ntoh(buffer[i]), length(buffer))

@inline function reconstruct_mat_bmw(buffer::MVector{12}, ::Type{T}) where {T}
    zer = zero(Complex{T})
    M₁₁ = Complex{T}(buffer[1], buffer[2])
    M₁₂ = Complex{T}(buffer[3], buffer[4])
    M₁₃ = Complex{T}(buffer[5], buffer[6])
    M₂₁ = Complex{T}(buffer[7], buffer[8])
    M₂₂ = Complex{T}(buffer[9], buffer[10])
    M₂₃ = Complex{T}(buffer[11], buffer[12])
    return SMatrix{3,3,Complex{T}}(M₁₁, M₂₁, zer, M₁₂, M₂₂, zer, M₁₃, M₂₃, zer)
end

@inline function deconstruct_mat_bmw(mat::SMatrix{3,3,Complex{T},9}) where {T}
    tup = (
        Float64(real(mat[1, 1])),
        Float64(imag(mat[1, 1])),
        Float64(real(mat[1, 2])),
        Float64(imag(mat[1, 2])),
        Float64(real(mat[1, 3])),
        Float64(imag(mat[1, 3])),
        Float64(real(mat[2, 1])),
        Float64(imag(mat[2, 1])),
        Float64(real(mat[2, 2])),
        Float64(imag(mat[2, 2])),
        Float64(real(mat[2, 3])),
        Float64(imag(mat[2, 3])),
    )
    return SVector{12,Float64}(tup)
end


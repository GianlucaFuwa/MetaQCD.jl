"""
    Liefield(NX, NY, NZ, NT)
    Liefield(u::Abstractfield)

Creates a Liefield, i.e. an array of 3-by-3 matrices of size `4 × NX × NY × NZ × NT`
or of the same size as `u`. Essentially the same as `TemporaryField`, but used for verbosity
in HMC
"""
struct Liefield <: Abstractfield
    U::Vector{Array{SMatrix{3,3,ComplexF64,9},4}}
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NV::Int64
    NC::Int64

    function Liefield(NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        U = Vector{Array{SMatrix{3,3,ComplexF64,9},4}}(undef, 4)

        for μ in 1:4
            U[μ] = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, NX, NY, NZ, NT)
        end

        return new(U, NX, NY, NZ, NT, NV, 3)
    end
end

Liefield(u::Abstractfield) = Liefield(u.NX, u.NY, u.NZ, u.NT)

function gaussian_momenta!(p::Liefield, ϕ)
    cosϕ = cos(ϕ)
    sinϕ = sin(ϕ)

    @threads for site in eachindex(p)
        for μ in 1:4
            p[μ][site] = cosϕ*p[μ][site] + sinϕ*gaussian_su3_matrix()
        end
    end

    return nothing
end

function flip_momenta!(p::Liefield)
    @batch for site in eachindex(p)
        for μ in 1:4
            p[μ][site] *= -1
        end
    end

    return nothing
end

function calc_kinetic_energy(p::Liefield)
    kin = 0.0
    @batch threadlocal=0.0::Float64 for site in eachindex(p)
        for μ in 1:4
            threadlocal += real(multr(p[μ][site], p[μ][site]))
        end
    end
    kin += sum(threadlocal)
    return kin
end

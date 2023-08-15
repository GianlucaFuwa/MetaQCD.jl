# Using mathematical (traceless-antihermitian) definition of 𝔰𝔲(3)
# This struct is not needed as we have Temporaryfield already, but it's nice for verbosity
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

function gaussian_momenta!(p::Liefield)
    @threads for site in eachindex(p)
        for μ in 1:4
            p[μ][site] = gaussian_su3_matrix()
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

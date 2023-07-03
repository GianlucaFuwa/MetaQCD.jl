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

        for μ = 1:4
            U[μ] = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, NX, NY, NZ, NT)
        end

        return new(U, NX, NY, NZ, NT, NV, 3)
    end

    function Liefield(u::T) where {T <: Abstractfield}
        return Liefield(u.NX, u.NY, u.NZ, u.NT)
    end
end

function gaussian_momenta!(p::Liefield)
    @batch for site in eachindex(p)
        for μ in 1:4
            p[μ][site] = gaussian_su3_matrix()
        end
    end

    return nothing
end

function calc_kinetic_energy(p::Liefield)
    spacing = 8
    ekin = zeros(Float64, nthreads() * spacing)

    @batch for site in eachindex(p)
        for μ = 1:4
            ekin[threadid() * spacing] += real(multr(p[μ][site], p[μ][site]))
        end
    end

    return sum(ekin)
end

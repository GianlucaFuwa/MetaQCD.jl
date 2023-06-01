# Using mathematical (traceless-antihermitian) definition of 𝔰𝔲(3)
# This struct is not needed as we have TemporaryField already, but it's nice for verbosity
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
        U = Vector{Array{SMatrix{3,3,ComplexF64,9},4}}(undef, 0)

        for μ = 1:4
            Uμ = Array{SMatrix{3,3,ComplexF64,9},4}(undef, NX, NY, NZ, NT)
            fill!(Uμ, SMatrix{3,3,ComplexF64,9}(zeros(3, 3)))
            push!(U, Uμ)
        end

        return new(U, NX, NY, NZ, NT, NV, 3)
    end

    function Liefield(u::Abstractfield)
        return Liefield(u.NX, u.NY, u.NZ, u.NT)
    end
end

function gaussian_momenta!(p::Liefield)
    NX, NY, NZ, NT = size(p)
    
    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        p[μ][ix,iy,iz,it] = gaussian_su3_matrix()
                    end
                end
            end
        end
    end

    return nothing
end

function calc_kinetic_energy(p::Liefield)
    NX, NY, NZ, NT = size(p)
    H_kin = 0.0

    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for μ = 1:4
                        H_kin += 
                            real(multr(p[μ][ix,iy,iz,it], p[μ][ix,iy,iz,it]))
                    end
                end
            end
        end
    end 
    
    return H_kin
end
struct Liefield{BACKEND,T,A,GA} <: Abstractfield{BACKEND,T,A}
    U::A # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors
    function Liefield{BACKEND,T,GA}(NX, NY, NZ, NT) where {BACKEND,T,GA}
        U = KA.zeros(BACKEND(), SVector{8,T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        return new{BACKEND,T,typeof(U),GA}(U, NX, NY, NZ, NT, NV, 3)
    end
end

function gaussian_TA!(p::Liefield{CPU,T}, ϕ=0) where {T}
    # friction is a number in the range [0,1] instead of an angle; it's easier to use
    # have to make sure that ϕ₁² + ϕ₂² = 1
    ϕ₁ = T(sqrt(1 - ϕ^2))
    ϕ₂ = T(ϕ)

    for site in eachindex(p)
        for μ in 1:4
            p[μ, site] = ϕ₁ * gaussian_TA_mat(T) + ϕ₂ * p[μ, site]
        end
    end

    return nothing
end

function gaussian_TA!(p::Colorfield{CPU,T}, ϕ=0) where {T}
    # friction is a number in the range [0,1] instead of an angle; it's easier to use
    # have to make sure that ϕ₁² + ϕ₂² = 1
    ϕ₁ = T(sqrt(1 - ϕ^2))
    ϕ₂ = T(ϕ)

    for site in eachindex(p)
        for μ in 1:4
            p[μ, site] = ϕ₁ * gaussian_TA_mat(T) + ϕ₂ * p[μ, site]
        end
    end

    return nothing
end

function calc_kinetic_energy(p::Liefield{CPU})
    K = 0.0

    @batch reduction = (+, K) for site in eachindex(p)
        for μ in 1:4
            pmat = materialize_TA(p[μ, site]...)
            K += real(multr(pmat, pmat))
        end
    end

    return K
end

function calc_kinetic_energy(p::Colorfield{CPU})
    K = 0.0

    @batch reduction = (+, K) for site in eachindex(p)
        for μ in 1:4
            pmat = p[μ, site]
            K += real(multr(pmat, pmat))
        end
    end

    return K
end

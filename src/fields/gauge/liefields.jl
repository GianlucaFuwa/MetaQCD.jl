function gaussian_TA!(p::Temporaryfield{CPU,T}, ϕ) where {T}
    ϕ₁ = T(sqrt(1 - ϕ^2))
    ϕ₂ = T(ϕ)

    @batch for site in eachindex(p)
        for μ in 1:4
            p[μ,site] = ϕ₁*gaussian_TA_mat(T) + ϕ₂*p[μ,site]
        end
    end

    return nothing
end

function calc_kinetic_energy(p::Temporaryfield{CPU})
    K = 0.0

    @batch reduction=(+, K) for site in eachindex(p)
        for μ in 1:4
            pmat = p[μ,site]
            K += real(multr(pmat, pmat))
        end
    end

    return K
end

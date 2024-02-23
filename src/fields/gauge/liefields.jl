function gaussian_TA!(p::Temporaryfield{CPUD,T}, ϕ) where {T}
    ϕ₁ = T(sqrt(1 - ϕ^2))
    ϕ₂ = T(ϕ)

    @threads for site in eachindex(p)
        for μ in 1:4
            p[μ,site] = ϕ₁*gaussian_TA_mat(T) + ϕ₂*p[μ,site]
        end
    end

    return nothing
end

function calc_kinetic_energy(p::Temporaryfield{CPUD})
    out = zeros(Float64, 8, nthreads())

    @threads for site in eachindex(p)
        for μ in 1:4
            pmat = p[μ,site]
            out[1,threadid()] += real(multr(pmat, pmat))
        end
    end

    return sum(out)
end

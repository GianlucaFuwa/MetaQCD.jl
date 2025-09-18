function gaussian_TA!(p::Colorfield{B,T,M}, ϕ=0.0) where {B,T,M}
    # friction is a number in the range [0,1] instead of an angle; it's easier to use
    # have to make sure that ϕ₁² + ϕ₂² = 1
    ϕ₁ = T(sqrt(1 - ϕ^2))
    ϕ₂ = T(ϕ)

    parallelfor(allindices(p), B, Val(M), (), (p,), (p,)) do μsite, (p,)
        p[μsite] = ϕ₁ * gaussian_TA_mat(T) + ϕ₂ * p[μsite]
    end

    return nothing
end

function calc_kinetic_energy(p::Colorfield{B,T,M}) where {B,T,M}
    K = parallelfor_sum(allindices(p), 0.0, B, Val(M), (), (), (p,)) do kₙ, μsite, (p,)
        pmat = p[μsite]
        kₙ += real(multr(pmat, pmat))
    end

    return distributed_reduce(K, +, p)
end

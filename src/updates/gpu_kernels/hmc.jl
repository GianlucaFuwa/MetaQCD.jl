function updateU!(
    U::Gaugefield{B,T}, hmc::HMC, fac, fermion_action, bias, therm, level
) where {B<:GPU,T}
    if level == 1
        ϵ = T(hmc.levels[level].Δτ * fac)
        P = hmc.P
        @latmap(eachindex(U, P), updateU_gpu!, U, P, ϵ)
    else
        evolve!(U, hmc, fermion_action, bias, therm, level-1)
    end
    return nothing
end

@kernel cpu=false function updateU_gpu!(U, @Const(P), ϵ, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in 1i32:4i32
        @inbounds U[μ,site] = cmatmul_oo(exp_iQ(-im * ϵ * P[μsite]), U[μsite])
    end
end

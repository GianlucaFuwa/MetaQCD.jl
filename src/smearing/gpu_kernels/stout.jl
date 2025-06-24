function apply_stout_smearing!(
    Uout::Gaugefield{B,T}, C::Colorfield{B,T}, Q::Expfield{B,T}, U::Gaugefield{B,T}, ρ
) where {B<:GPU,T}
    bulk = eachindex(Uout, U, C, Q)
    # TODO: can hide
    update_halo!(U)
    @latmap(bulk, apply_stout_smearing_gpu!, Uout, C, Q, U, T(ρ))
    return nothing
end

@kernel cpu=false function apply_stout_smearing_gpu!(Uout, C, Q, @Const(U), ρ, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in 1i32:4i32
        Qμ = calc_stout_Q_kernel!(Q, C, U, site, μ, ρ)
        @inbounds Uout[μ, site] = cmatmul_oo(exp_iQ(Qμ), U[μ, site])
    end
end

function stout_recursion!(
    Σ::Colorfield{B,T},
    Σ′::Colorfield{B,T},
    U′::Gaugefield{B,T},
    U::Gaugefield{B,T},
    C::Colorfield{B,T},
    Q::Expfield{B,T},
    Λ::Colorfield{B,T},
    ρ,
) where {B<:GPU,T}
    leftmul_dagg!(Σ′, U′)
    calc_stout_Λ!(Λ, Σ′, Q, U)
    bulk = eachindex(U, Σ, Σ′, U′, C, Q, Λ)
    # TODO: can hide
    update_halo!(U, Λ)
    @latmap(bulk, stout_recursion_gpu!, Σ, Σ′, U, C, Q, Λ, T(ρ))
    return nothing
end

@kernel cpu=false function stout_recursion_gpu!(
    Σ, @Const(Σ′), @Const(U), @Const(C), @Const(Q), @Const(Λ), ρ, bulk
)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in 1i32:4i32
        stout_recursion_kernel!(Σ, Σ′, U, C, Q, Λ, site, μ, ρ)
    end
end

function calc_stout_Λ!(
    Λ::Colorfield{B,T}, Σ′::Colorfield{B,T}, Q::Expfield{B,T}, U::Gaugefield{B,T}
) where {B<:GPU,T}
    bulk = eachindex(U ,Λ, Σ′, Q)
    @latmap(bulk, calc_stout_Λ_gpu!, Λ, Σ′, Q, U)
    return nothing
end

@kernel cpu=false function calc_stout_Λ_gpu!(Λ, @Const(Σ′), @Const(Q), @Const(U), bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in 1i32:4i32
        calc_stout_Λ_kernel!(Λ, Σ′, Q, U, site, μ)
    end
end

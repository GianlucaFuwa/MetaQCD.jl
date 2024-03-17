function apply_stout_smearing!(
    Uout::Gaugefield{B,T}, C::Temporaryfield{B,T}, Q::CoeffField{B,T}, U::Gaugefield{B}, ρ
) where {B<:GPU,T}
    @assert dims(Uout) == dims(U) == dims(C) == dims(Q)
    @latmap(Sequential(), Val(1), apply_stout_smearing_kernel!, Uout, C, Q, U, T(ρ))
    return nothing
end

@kernel function apply_stout_smearing_kernel!(Uout, C, Q, @Const(U), ρ)
    site = @index(Global, Cartesian)

    @unroll for μ in 1i32:4i32
        Qμ = calc_stout_Q!(Q, C, U, site, μ, ρ)
        @inbounds Uout[μ, site] = cmatmul_oo(exp_iQ(Qμ), U[μ, site])
    end
end

function stout_recursion!(
    Σ::Temporaryfield{B,T},
    Σ′::Temporaryfield{B},
    U′::Gaugefield{B},
    U::Gaugefield{B},
    C::Temporaryfield{B},
    Q::CoeffField{B},
    Λ::Temporaryfield{B},
    ρ,
) where {B<:GPU,T}
    @assert dims(U) == dims(Σ) == dims(Σ′) == dims(U′) == dims(C) == dims(Q) == dims(Λ)
    leftmul_dagg!(Σ′, U′)
    calc_stout_Λ!(Λ, Σ′, Q, U)
    @latmap(Sequential(), Val(1), stout_recursion_kernel!, Σ, Σ′, U, C, Q, Λ, T(ρ))
    return nothing
end

@kernel function stout_recursion_kernel!(
    Σ, @Const(Σ′), @Const(U), @Const(C), @Const(Q), @Const(Λ), ρ
)
    site = @index(Global, Cartesian)
    dimsΣ′ = dims(Σ′)

    @unroll for μ in 1i32:4i32
        @inbounds Nμ = dimsΣ′[μ]
        siteμp = move(site, μ, 1i32, Nμ)
        force_sum = zero3(float_type(Σ))

        for ν in 1i32:4i32
            if ν == μ
                continue
            end

            Nν = dimsΣ′[ν]
            siteνp = move(site, ν, 1i32, Nν)
            siteνn = move(site, ν, -1i32, Nν)
            siteμpνn = move(siteμp, ν, -1i32, Nν)

            # bring reused matrices up to cache (can also precalculate some products)
            # Uνsiteμ⁺ = U[ν,siteμp]
            # Uμsiteμ⁺ = U[μ,siteνp]
            # Uνsite = U[ν,site]
            # Uνsiteμ⁺ν⁻ = U[ν,siteμpνn]
            # Uμsiteν⁻ = U[μ,siteνn]
            # Uνsiteν⁻ = U[ν,siteνn]

            @inbounds force_sum +=
                cmatmul_oddo(U[ν, siteμp], U[μ, siteνp], U[ν, site], Λ[ν, site]) +
                cmatmul_ddoo(U[ν, siteμpνn], U[μ, siteνn], Λ[μ, siteνn], U[ν, siteνn]) +
                cmatmul_dodo(U[ν, siteμpνn], Λ[ν, siteμpνn], U[μ, siteνn], U[ν, siteνn]) -
                cmatmul_ddoo(U[ν, siteμpνn], U[μ, siteνn], Λ[ν, siteνn], U[ν, siteνn]) -
                cmatmul_oodd(Λ[ν, siteμp], U[ν, siteμp], U[μ, siteνp], U[ν, site]) +
                cmatmul_odod(U[ν, siteμp], U[μ, siteνp], Λ[μ, siteνp], U[ν, site])
        end

        @inbounds begin
            link = U[μ, site]
            expiQ_mat = exp_iQ(Q[μ, site])
            Σ[μ, site] = traceless_antihermitian(
                cmatmul_ooo(link, Σ′[μ, site], expiQ_mat) +
                im * cmatmul_odo(link, C[μ, site], Λ[μ, site]) -
                im * ρ * cmatmul_oo(link, force_sum),
            )
        end
    end
end

function calc_stout_Λ!(
    Λ::Temporaryfield{B}, Σ′::Temporaryfield{B}, Q::CoeffField{B}, U::Gaugefield{B}
) where {B<:GPU}
    @assert dims(U) == dims(Λ) == dims(Σ′) == dims(Q)
    @latmap(Sequential(), Val(1), calc_stout_Λ_kernel!, Λ, Σ′, Q, U)
    return nothing
end

@kernel function calc_stout_Λ_kernel!(Λ, @Const(Σ′), @Const(Q), @Const(U))
    site = @index(Global, Cartesian)

    @unroll for μ in 1i32:4i32
        @inbounds q = Q[μ, site]
        Qₘ = get_Q(q)
        Q² = get_Q²(q)
        @inbounds UΣ′ = cmatmul_oo(U[μ, site], Σ′[μ, site])

        B₁ = get_B₁(q)
        B₂ = get_B₂(q)

        Γ =
            multr(B₁, UΣ′) * Qₘ +
            multr(B₂, UΣ′) * Q² +
            q.f₁ * UΣ′ +
            q.f₂ * cmatmul_oo(Qₘ, UΣ′) +
            q.f₂ * cmatmul_oo(UΣ′, Qₘ)

        @inbounds Λ[μ, site] = traceless_hermitian(Γ)
    end
end

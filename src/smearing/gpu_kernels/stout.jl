function apply_stout_smearing!(Uout::Gaugefield{GPUD,T}, C::Temporaryfield{GPUD,T},
    Q::CoeffField{GPUD,T}, U::Gaugefield{GPUD}, ρ) where {T}
    @assert size(Uout) == size(U) == size(C)== size(Q)
    @latmap(apply_stout_smearing_kernel!, Uout, C, Q, U, T(ρ))
    return nothing
end

@kernel function apply_stout_smearing_kernel!(Uout, C, Q, @Const(U), ρ)
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
		Qμ = calc_stout_Q!(Q, C, U, site, μ, ρ)
        @inbounds Uout[μ,site] = cmatmul_oo(exp_iQ(Qμ), U[μ,site])
	end
end

function stout_recursion!(Σ::Temporaryfield{GPUD,T}, Σ′::Temporaryfield{GPUD},
    U′::Gaugefield{GPUD}, U::Gaugefield{GPUD}, C::Temporaryfield{GPUD},
    Q::CoeffField{GPUD}, Λ::Temporaryfield{GPUD}, ρ) where {T}
    @assert size(U) == size(Σ) == size(Σ′) == size(U′) == size(C) == size(Q) == size(Λ)
    @latmap(stout_recursion_kernel!, Σ, Σ′, U′, U, C, Q, Λ, T(ρ))
    return nothing
end

@kernel function stout_recursion_kernel!(Σ, @Const(Σ′), @Const(U), @Const(C), @Const(Q),
    @Const(Λ), ρ)
	site = @index(Global, Cartesian)

	@inbounds for μ in 1i32:4i32
		Nμ = sizeΣ′[μ+1i32]
        siteμp = move(site, μ, 1i32, Nμ)
        force_sum = zero3

        for ν in 1i32:4i32
            if ν == μ
                continue
            end

            Nν = sizeΣ′[ν+1i32]
            siteνp = move(site, ν, 1i32, Nν)
            siteνn = move(site, ν, -1i32 ,Nν)
            siteμpνn = move(siteμp, ν, -1i32, Nν)

            # bring reused matrices up to cache (can also precalculate some products)
            # Uνsiteμ⁺ = U[ν,siteμp]
            # Uμsiteμ⁺ = U[μ,siteνp]
            # Uνsite = U[ν,site]
            # Uνsiteμ⁺ν⁻ = U[ν,siteμpνn]
            # Uμsiteν⁻ = U[μ,siteνn]
            # Uνsiteν⁻ = U[ν,siteνn]

            force_sum +=
                cmatmul_oddo(U[ν,siteμp]  , U[μ,siteνp]  , U[ν,site]  , Λ[ν,site])   +
                cmatmul_ddoo(U[ν,siteμpνn], U[μ,siteνn]  , Λ[μ,siteνn], U[ν,siteνn]) +
                cmatmul_dodo(U[ν,siteμpνn], Λ[ν,siteμpνn], U[μ,siteνn], U[ν,siteνn]) -
                cmatmul_ddoo(U[ν,siteμpνn], U[μ,siteνn]  , Λ[ν,siteνn], U[ν,siteνn]) -
                cmatmul_oodd(Λ[ν,siteμp]  , U[ν,siteμp]  , U[μ,siteνp], U[ν,site])   +
                cmatmul_odod(U[ν,siteμp]  , U[μ,siteνp]  , Λ[μ,siteνp], U[ν,site])
        end

        link = U[μ,site]
        expiQ_mat = exp_iQ(Q[μ,site])
        Σ[μ,site] = traceless_antihermitian(cmatmul_ooo(link, Σ′[μ,site], expiQ_mat) +
                                            im*cmatmul_odo(link, C[μ,site], Λ[μ,site]) -
                                            im*ρ*cmatmul_oo(link, force_sum))
	end
end

function calc_stout_Λ!(Λ::Temporaryfield{GPUD}, Σ′::Temporaryfield{GPUD},
    Q::CoeffField{GPUD}, U::Gaugefield{GPUD})
    @assert size(U) == size(Λ) == size(Σ′) == size(Q)
    @latmap(calc_stout_Λ_kernel!, Λ, Σ′, Q, U)
    return nothing
end

@kernel function calc_stout_Λ_kernel!(Λ, @Const(Σ′), @Const(Q), @Const(U))
	site = @index(Global, Cartesian)

	@unroll for μ in 1i32:4i32
		@inbounds q = Q[μ,site]
        Qₘ = get_Q(q)
        Q² = get_Q²(q)
        @inbounds UΣ′ = cmatmul_oo(U[μ,site], Σ′[μ,site])

        B₁ = get_B₁(q)
        B₂ = get_B₂(q)

        Γ = multr(B₁, UΣ′) * Qₘ +
            multr(B₂, UΣ′) * Q² +
            q.f₁ * UΣ′ +
            q.f₂ * cmatmul_oo(Qₘ, UΣ′) +
            q.f₂ * cmatmul_oo(UΣ′, Qₘ)

        @inbounds Λ[μ,site] = traceless_hermitian(Γ)
	end
end

import ..Gaugefields: WilsonGaugeAction
"""
Struct StoutSmearing holds all fields relevant to smearing and subsequent recursion. \\
Since we never actually use the smeared fields in main, they dont have to leave this scope
"""
struct StoutSmearing{TG} <: AbstractSmearing
	numlayers::Int64
	ρ::Float64
	Usmeared_multi::Vector{TG}
	C_multi::Vector{Temporaryfield}
	Q_multi::Vector{CoeffField}
	Λ::Temporaryfield

	function StoutSmearing(U::TG, numlayers, ρ) where {TG}
		@assert numlayers >= 0 && ρ >= 0 "number of stout layers and ρ must be >= 0"

		if numlayers == 0 || ρ == 0
			return NoSmearing()
		else
			Usmeared_multi = Vector{TG}(undef, numlayers + 1)
			C_multi = Vector{Temporaryfield}(undef, numlayers)
			Q_multi = Vector{CoeffField}(undef, numlayers)
			Λ = Temporaryfield(U)

			Usmeared_multi[1] = similar(U)

			for i in 1:numlayers
				Usmeared_multi[i+1] = similar(U)
				C_multi[i] = Temporaryfield(U)
				Q_multi[i] = CoeffField(U)
			end

			return new{TG}(numlayers, ρ, Usmeared_multi, C_multi, Q_multi, Λ)
		end
	end
end

function Base.length(s::T) where {T <: StoutSmearing}
	return s.numlayers
end

function get_layer(s::T, i) where {T <: StoutSmearing}
	return s.Usmeared_multi[i]
end

function apply_smearing!(smearing, Uin)
	numlayers = length(smearing)
	ρ = smearing.ρ
	Uout_multi = smearing.Usmeared_multi
	C_multi = smearing.C_multi
	Q_multi = smearing.Q_multi

	substitute_U!(Uout_multi[1], Uin)

	for i in 1:numlayers
		apply_stout_smearing!(Uout_multi[i+1], C_multi[i], Q_multi[i], Uout_multi[i], ρ)
	end

	return nothing
end

function apply_stout_smearing!(Uout, C, Q, U, ρ)
	calc_stout_Q!(Q, C, U, ρ)

	@batch for site in eachindex(U)
        for μ in 1:4
            Uout[μ][site] = cmatmul_oo(exp_iQ(Q[μ][site]), U[μ][site])
        end
	end

	return nothing
end

function stout_backprop!(Σ_current, Σ_prev, smearing)
	ρ = smearing.ρ
	numlayers = length(smearing)

	Usmeared_multi = smearing.Usmeared_multi
	C_multi = smearing.C_multi
	Q_multi = smearing.Q_multi
	Λ = smearing.Λ

	for i in reverse(1:numlayers)
		stout_recursion!(
			Σ_prev,
			Σ_current,
			Usmeared_multi[i+1],
			Usmeared_multi[i],
			C_multi[i],
			Q_multi[i],
			Λ,
			ρ,
		)
		substitute_U!(Σ_current, Σ_prev)
	end

	return nothing
end

"""
Stout-Force recursion \\
See: hep-lat/0311018 by Morningstar & Peardon \\
\\
Σμ = Σμ'⋅exp(iQμ) + iCμ†⋅Λμ - i∑{...}
"""
function stout_recursion!(Σ, Σ′, U′, U, C, Q, Λ, ρ)
	leftmul_dagg!(Σ′, U′)
	calc_stout_Λ!(Λ, Σ′, Q, U)

	@batch for site in eachindex(Σ)
        for μ in 1:4
            Nμ = size(Σ′)[μ]
            siteμp = move(site, μ, 1, Nμ)
            force_sum = @SMatrix zeros(ComplexF64, 3, 3)

            for ν in 1:4
                if ν == μ
                    continue
                end

                Nν = size(Σ′)[ν]
                siteνp = move(site, ν, 1, Nν)
                siteνn = move(site, ν, -1 ,Nν)
                siteμpνn = move(siteμp, ν, -1, Nν)

                force_sum +=
                    cmatmul_oddo(U[ν][siteμp], U[μ][siteνp], U[ν][site], Λ[ν][site]) +
                    cmatmul_ddoo(U[ν][siteμpνn], U[μ][siteνn], Λ[μ][siteνn], U[ν][siteνn]) +
                    cmatmul_dodo(U[ν][siteμpνn], Λ[ν][siteμpνn], U[μ][siteνn], U[ν][siteνn]) -
                    cmatmul_ddoo(U[ν][siteμpνn], U[μ][siteνn], Λ[ν][siteνn], U[ν][siteνn]) -
                    cmatmul_oodd(Λ[ν][siteμp], U[ν][siteμp], U[μ][siteνp], U[ν][site]) +
                    cmatmul_odod(U[ν][siteμp], U[μ][siteνp], Λ[μ][siteνp], U[ν][site])
            end

            link = U[μ][site]
            expiQ_mat = exp_iQ(Q[μ][site])
            Σ[μ][site] = traceless_antihermitian(
                cmatmul_ooo(link, Σ′[μ][site], expiQ_mat) +
                im * cmatmul_odo(link, C[μ][site], Λ[μ][site]) -
                im * ρ * cmatmul_oo(link, force_sum)
            )
        end
	end

	return nothing
end

"""
Γ = Tr(Σ'⋅B1⋅U)⋅Q + Tr(Σ'⋅B2⋅U)⋅Q²
	+ f1⋅U⋅Σ' + f2⋅Q⋅U⋅Σ' + f1⋅U⋅Σ'⋅Q \\
Λ = 1/2⋅(Γ + Γ†) - 1/(2N)⋅Tr(Γ + Γ†)
"""
function calc_stout_Λ!(Λ, Σ′, Q, U)
	@batch for site in eachindex(Λ)
        for μ in 1:4
            q = Q[μ][site]
            Q_mat = q.Q
            Q2_mat = q.Q2
            UΣ′ = cmatmul_oo(U[μ][site], Σ′[μ][site])

            B1_mat = B1(q)
            B2_mat = B2(q)

            Γ = multr(B1_mat, UΣ′) * Q_mat +
                multr(B2_mat, UΣ′) * Q2_mat +
                q.f1 * UΣ′ +
                q.f2 * cmatmul_oo(Q_mat, UΣ′) +
                q.f2 * cmatmul_oo(UΣ′, Q_mat)

            Λ[μ][site] = traceless_hermitian(Γ)
        end
	end

	return nothing
end

function calc_stout_Q!(Q, C, U, ρ)
	@batch for site in eachindex(Q)
        for μ in 1:4
            Cμ = ρ * staple(WilsonGaugeAction(), U, μ, site)
            C[μ][site] = Cμ

            Ω = cmatmul_od(Cμ, U[μ][site])
            Q[μ][site] = exp_iQ_coeffs(-im * traceless_antihermitian(Ω))
        end
	end

	return nothing
end

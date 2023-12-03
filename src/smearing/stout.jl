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
		@assert numlayers>=0 && ρ>=0 "number of stout layers and ρ must be >= 0"

		if numlayers==0 || ρ==0
			return NoSmearing()
		else
			Usmeared_multi = Vector{TG}(undef, numlayers+1)
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

function Base.length(s::T) where {T<:StoutSmearing}
	return s.numlayers
end

function get_layer(s::T, i) where {T<:StoutSmearing}
	return s.Usmeared_multi[i]
end

function apply_smearing!(smearing, Uin)
	numlayers = length(smearing)
	ρ = smearing.ρ
	Usmeared = smearing.Usmeared_multi
	C = smearing.C_multi
	Q = smearing.Q_multi

	substitute_U!(Usmeared[1], Uin)

	for i in 1:numlayers
		apply_stout_smearing!(Usmeared[i+1], C[i], Q[i], Usmeared[i], ρ)
	end

	return nothing
end

function apply_stout_smearing!(Uout, C, Q, U, ρ)
	@batch per=thread for site in eachindex(U)
        for μ in 1:4
			Qμ = calc_stout_Q!(Q, C, U, site, μ, ρ)
            Uout[μ][site] = cmatmul_oo(exp_iQ(Qμ), U[μ][site])
        end
	end

	return nothing
end

function stout_backprop!(Σ′, Σ, smearing)
	Usmeared = smearing.Usmeared_multi
	C = smearing.C_multi
	Q = smearing.Q_multi
	Λ = smearing.Λ

	for i in reverse(1:length(smearing))
		stout_recursion!(Σ, Σ′, Usmeared[i+1], Usmeared[i], C[i], Q[i], Λ, smearing.ρ)
		substitute_U!(Σ′, Σ)
	end

	return nothing
end

"""
Stout-Force recursion \\
See [hep-lat/0311018] by Morningstar & Peardon
"""
function stout_recursion!(Σ, Σ′, U′, U, C, Q, Λ, ρ)
	leftmul_dagg!(Σ′, U′)
	calc_stout_Λ!(Λ, Σ′, Q, U)
	sizeΣ′ = size(Σ′)

	@batch per=thread for site in eachindex(Σ)
        for μ in 1:4
            Nμ = sizeΣ′[μ]
            siteμp = move(site, μ, 1, Nμ)
            force_sum = zero3

            for ν in 1:4
                if ν == μ
                    continue
                end

                Nν = sizeΣ′[ν]
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
            Σ[μ][site] = traceless_antihermitian(cmatmul_ooo(link, Σ′[μ][site], expiQ_mat) +
                								 im*cmatmul_odo(link, C[μ][site], Λ[μ][site]) -
                								 im*ρ*cmatmul_oo(link, force_sum))
        end
	end

	return nothing
end

function calc_stout_Λ!(Λ, Σ′, Q, U)
	@batch per=thread for site in eachindex(Λ)
        for μ in 1:4
			q = Q[μ][site]
			Qₘ = get_Q(q)
			Q² = get_Q²(q)
			UΣ′ = cmatmul_oo(U[μ][site], Σ′[μ][site])

			B₁ = get_B₁(q)
			B₂ = get_B₂(q)

			Γ = multr(B₁, UΣ′) * Qₘ +
				multr(B₂, UΣ′) * Q² +
				q.f₁ * UΣ′ +
				q.f₂ * cmatmul_oo(Qₘ, UΣ′) +
				q.f₂ * cmatmul_oo(UΣ′, Qₘ)

			Λ[μ][site] = traceless_hermitian(Γ)
		end
	end

	return nothing
end

function calc_stout_Q!(Q, C, U, site, μ, ρ)
	Cμ = ρ * staple(WilsonGaugeAction(), U, μ, site)
	C[μ][site] = Cμ

	Ω = cmatmul_od(Cμ, U[μ][site])
	Qμ = exp_iQ_coeffs(-im * traceless_antihermitian(Ω))
	Q[μ][site] = Qμ
	return Qμ
end

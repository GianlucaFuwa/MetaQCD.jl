import ..Gaugefields: WilsonGaugeAction
"""
Struct StoutSmearing holds all fields relevant to smearing and subsequent recursion. \\
Since we never actually use the smeared fields in main, they dont have to leave this scope
"""
struct StoutSmearing{TG} <: AbstractSmearing
	numlayers::Int64
	ρ::Float64
	Usmeared_multi::Vector{Gaugefield{TG}}
	C_multi::Vector{Temporaryfield}
	Q_multi::Vector{CoeffField}
	Λ::Temporaryfield

	function StoutSmearing(
		U::Gaugefield{TG},
		numlayers,
		ρ,
	) where {TG <: AbstractGaugeAction}
		@assert numlayers >= 0 && ρ >= 0 "number of stout layers and ρ must be >= 0"
		
		if numlayers == 0 || ρ == 0
			return NoSmearing()
		else
			Usmeared_multi = Vector{Gaugefield{TG}}(undef, numlayers + 1)
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

function apply_smearing!(
	smearing,
	Uin,
)
	numlayers = length(smearing)
	ρ = smearing.ρ
	Uout_multi = smearing.Usmeared_multi
	C_multi = smearing.C_multi
	Q_multi = smearing.Q_multi

	substitute_U!(Uout_multi[1], Uin)

	for i in 1:numlayers
		apply_stout_smearing!(
			Uout_multi[i+1],
			C_multi[i],
			Q_multi[i],
			Uout_multi[i],
			ρ,
		)
	end

	return nothing
end

function apply_stout_smearing!(
	Uout,
	C,
	Q,
	U,
	ρ,
)
	NX, NY, NZ, NT = size(Uout)
	calc_stout_Q!(Q, C, U, ρ)
	
	@batch for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					for μ in 1:4
						@inbounds Uout[μ][ix,iy,iz,it] = 
							exp_iQ(Q[μ][ix,iy,iz,it]) * U[μ][ix,iy,iz,it]
					end
				end
			end
		end
	end
	
	return nothing
end

function stout_backprop!(
	Σ_current,
	Σ_prev,
	smearing,
)
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
function stout_recursion!(
	Σ,
	Σ_prime,
	U_prime,
	U,
	C,
	Q,
	Λ,
	ρ,
)
	NX, NY, NZ, NT = size(Σ_prime)

	leftmul!(adjoint, Σ_prime, U_prime)

	calc_stout_Λ!(Λ, Σ_prime, Q, U)

	@batch for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					site = SiteCoords(ix, iy, iz, it)

					@inbounds for μ in 1:4
						Nμ = size(Σ_prime)[μ]
						siteμp = move(site, μ, 1, Nμ)
						force_sum = @SMatrix zeros(ComplexF64, 3, 3)

						for ν in 1:4
							if ν == μ
								continue
							end

							Nν = size(Σ_prime)[ν]
							siteνp = move(site, ν, 1, Nν)
							siteνn = move(site, ν, -1 ,Nν)
							siteμpνn = move(siteμp, ν, -1, Nν)

							force_sum += 
								U[ν][siteμp]    * U[μ][siteνp]'  * U[ν][site]'   * Λ[ν][site]   + 
								U[ν][siteμpνn]' * U[μ][siteνn]'  * Λ[μ][siteνn]  * U[ν][siteνn] + 
								U[ν][siteμpνn]' * Λ[ν][siteμpνn] * U[μ][siteνn]' * U[ν][siteνn] - 
								U[ν][siteμpνn]' * U[μ][siteνn]'  * Λ[ν][siteνn]  * U[ν][siteνn] - 
								Λ[ν][siteμp]    * U[ν][siteμp]   * U[μ][siteνp]' * U[ν][site]'  + 
								U[ν][siteμp]    * U[μ][siteνp]'  * Λ[μ][siteνp]  * U[ν][site]'
							
						end

						link = U[μ][site]
						expiQ_mat = exp_iQ(Q[μ][site])
						Σ[μ][site] = traceless_antihermitian(
							link * Σ_prime[μ][site] * expiQ_mat +
							im * link * C[μ][site]' * Λ[μ][site] -
							im * ρ * link * force_sum
						)
					end
				end
			end
		end
	end

	return nothing
end

"""
Γ = Tr(Σ'⋅B1⋅U)⋅Q + Tr(Σ'⋅B2⋅U)⋅Q² 
	+ f1⋅U⋅Σ' + f2⋅Q⋅U⋅Σ' + f1⋅U⋅Σ'⋅Q \\
Λ = 1/2⋅(Γ + Γ†) - 1/(2N)⋅Tr(Γ + Γ†)
"""
function calc_stout_Λ!(
	Λ,
	Σprime,
	Q,
	U,
)
	NX, NY, NZ, NT = size(U)

	@batch for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					@inbounds for μ in 1:4
						q = Q[μ][ix,iy,iz,it]
						Q_mat = q.Q
						UΣ = U[μ][ix,iy,iz,it] * Σprime[μ][ix,iy,iz,it]
						
						B1_mat = B1(q)
						B2_mat = B2(q)

						Γ = multr(B1_mat, UΣ) * Q_mat + 
							multr(B2_mat, UΣ) * Q_mat^2 +
							q.f1 * UΣ + 
							q.f2 * Q_mat * UΣ + 
							q.f2 * UΣ * Q_mat

						Λ[μ][ix,iy,iz,it] = traceless_hermitian(Γ)
					end
				end
			end
		end
	end

	return nothing
end

function calc_stout_Q!(
	Q,
	C,
	U,
	ρ,
)
	NX, NY, NZ, NT = size(U)
	staple = WilsonGaugeAction()

	@batch for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					site = SiteCoords(ix, iy, iz, it)

					@inbounds for μ in 1:4
						Cμ = ρ * staple(U, μ, site)
						C[μ][ix,iy,iz,it] = Cμ

						Ω = Cμ * U[μ][ix,iy,iz,it]'
						Q[μ][ix,iy,iz,it] = exp_iQ_coeffs(-im * traceless_antihermitian(Ω))
					end
					
				end
			end
		end
	end

	return nothing
end
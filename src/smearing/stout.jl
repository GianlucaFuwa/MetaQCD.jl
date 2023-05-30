struct StoutSmearing{T} <: AbstractSmearing
	numlayers::Int64
	ρ::Float64
	Usmeared_multi::Vector{Gaugefield{T}}
	C_multi::Vector{TemporaryField}
	Q_multi::Vector{CoeffField}
	Λ::TemporaryField

	function StoutSmearing(
		U::Gaugefield{T},
		numlayers,
		ρ;
		recursive = true,
	) where {T <: AbstractGaugeAction}
		@assert numlayers >= 0 && ρ >= 0 "number of stout layers and ρ must be >= 0"
		if numlayers == 0 || ρ == 0
			return NoSmearing()
		else
			Usmeared_multi = Vector{Gaugefield{T}}(undef, numlayers + 1)
			C_multi = Vector{TemporaryField}(undef, numlayers)
			Q_multi = Vector{CoeffField}(undef, numlayers)
			Λ = TemporaryField(U)
	
			Usmeared_multi[1] = similar(U)
	
			for i in 1:numlayers
				Usmeared_multi[i+1] = similar(U)
				C_multi[i] = TemporaryField(U)
				Q_multi[i] = CoeffField(U)
			end
	
			return new{T}(numlayers, ρ, Usmeared_multi, C_multi, Q_multi, Λ)
		end
	end
end

function Base.length(s::StoutSmearing{T}) where {T}
	return s.numlayers
end

function get_layer(s::StoutSmearing{T}, i) where {T}
	return s.Usmeared_multi[i]
end

function apply_smearing!(
	smearing::StoutSmearing{T},
	Uin::Gaugefield{T},
) where {T <: AbstractGaugeAction}
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
	Uout::Gaugefield{T},
	C::TemporaryField,
	Q::CoeffField,
	U::Gaugefield{T},
	ρ,
) where {T <: AbstractGaugeAction}
	NX, NY, NZ, NT = size(Uout)
	calc_stout_Q!(Q, C, U, ρ)
	
	for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					for μ in 1:4
						Uout[μ][ix,iy,iz,it] = exp_iQ(Q[μ][ix,iy,iz,it]) * U[μ][ix,iy,iz,it]
					end
				end
			end
		end
	end
	
	return nothing
end

function stout_backprop!(
	Σ_current::TemporaryField,
	Σ_prev::Union{Nothing, TemporaryField},
	# U_unsmeared::Gaugefield,
	smearing::StoutSmearing{T},
) where {T}
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
	Σ::TemporaryField,
	Σ_prime::TemporaryField,
	U_prime::Gaugefield{T},
	U::Gaugefield{T},
	C::TemporaryField,
	Q::CoeffField,
	Λ::TemporaryField,
	ρ,
) where {T <: AbstractGaugeAction}
	NX, NY, NZ, NT = size(Σ_prime)

	lmul!(adjoint, Σ_prime, U_prime)

	calc_stout_Λ!(Λ, Σ_prime, Q, U)

	for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					site = SiteCoords(ix, iy, iz, it)

					for μ in 1:4
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

						link = U[μ][ix,iy,iz,it]
						Q_mat = Q[μ][ix,iy,iz,it].Q
						Σ[μ][ix,iy,iz,it] = traceless_antihermitian(
							link * Σ_prime[μ][ix,iy,iz,it] * exp_iQ(Q_mat) +
							im * link * C[μ][ix,iy,iz,it]' * Λ[μ][ix,iy,iz,it] -
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
	Λ::TemporaryField,
	Σprime::TemporaryField,
	Q::CoeffField,
	U::Gaugefield{T},
) where {T <: AbstractGaugeAction}
	NX, NY, NZ, NT = size(U)

	for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					for μ in 1:4
						q = Q[μ][ix,iy,iz,it]
						Q_mat = q.Q
						UΣ = U[μ][ix,iy,iz,it] * Σprime[μ][ix,iy,iz,it]
						
						B1_mat = B1(q)
						B2_mat = B2(q)

						Γ = tr(B1_mat, UΣ) * Q_mat + 
							tr(B2_mat, UΣ) * Q_mat^2 +
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
	Q::CoeffField,
	C::TemporaryField,
	U::Gaugefield{T},
	ρ,
) where {T <: AbstractGaugeAction}
	NX, NY, NZ, NT = size(U)
	staple = T()

	for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					site = SiteCoords(ix, iy, iz, it)

					for μ in 1:4
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
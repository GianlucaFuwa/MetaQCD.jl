struct StoutSmearing <: AbstractSmearing
	numlayers::Int
	ρ::AbstractFloat
end

function Base.length(s::StoutSmearing)
	return s.numlayers
end

function get_ρ(s::StoutSmearing)
	return s.ρ
end

function apply_smearing(U::Gaugefield, smearing::StoutSmearing)
	numlayers = length(smearing)
	ρ = get_ρ(smearing)
	Uout_multi = Vector{T}(undef, numlayers)
	staples_multi = Vector{TemporaryField}(undef, numlayers-1)
	Qs_multi = Vector{TemporaryField}(undef, numlayers-1)

	for i in 1:numlayers
		Uout_multi[i] = similar(U)
		staples_multi[i] = TemporaryField(U)
		Qs_multi[i] = TemporaryField(U)
	end

	calc_stout_multi!(Uout_multi, staples_multi, Qs_multi, U, ρ, numlayers)
	return Uout_multi, staples_multi, Qs_multi
end

function calc_stout_multi!(
	Uout_multi::Vector{Gaugefield}, 
	staples_multi::Vector{TemporaryField}, 
	Qs_multi::Vector{TemporaryField}, 
	Uin::Gaugefield, 
	ρ, 
	numlayers,
) 
	U = deepcopy(Uin)

	for i in 1:numlayers
		substitute_U!(Uout_multi[i], U)
		apply_stout_smearing!(U, Uout_multi[i], staples_multi[i], Qs_multi[i], ρ)
	end

	substitute_U!(Uin, U)
	return nothing
end

function apply_stout_smearing!(
	Uout::Gaugefield,
	U::Gaugefield,
	staples::TemporaryField,
	Qs::TemporaryField,
	ρ,
)
	NX, NY, NZ, NT = size(Uout)
	staple_eachsite!(staples, U)
	calc_Qmatrices!(Qs, staples, U, ρ)
	
	@batch for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					Uout[1][ix,iy,iz,it] = exp_iQ(Qs[1][ix,iy,iz,it]) * U[1][ix,iy,iz,it]

					Uout[2][ix,iy,iz,it] = exp_iQ(Qs[2][ix,iy,iz,it]) * U[2][ix,iy,iz,it]

					Uout[3][ix,iy,iz,it] = exp_iQ(Qs[3][ix,iy,iz,it]) * U[3][ix,iy,iz,it]

					Uout[4][ix,iy,iz,it] = exp_iQ(Qs[4][ix,iy,iz,it]) * U[4][ix,iy,iz,it]
				end
			end
		end
	end
	
	return nothing
end

function stout_recursion!(
	Σcurrent::TemporaryField,
	Uout_multi::Union{Nothing, Vector{Gaugefield}},
	staples_multi::Union{Nothing, Vector{TemporaryField}},
	Qs_multi::Union{Nothing, Vector{TemporaryField}},
	smearing,
)
	if Uout_multi === nothing || staples_multi === nothing || Qs_multi === nothing
		return nothing
	end

	numlayers = length(smearing)
	Σprev = similar(Σcurrent)

	for i in numlayers:-1:1
		layer_backprop!(Σprev, Σcurrent, Uout_multi[i], staples_multi[i], Qs_multi[i], ρ)
		Σcurrent, Σprev = Σprev, Σcurrent
	end

	return nothing
end

"""
Stout-Force recursion \\
See: hep-lat/0311018 by Morningstar & Peardon \\
\\
Σμ = Σμ'⋅exp(iQμ) + iCμ†⋅Λμ - i∑{...}   
"""
function layer_backprop!(
	Σ::TemporaryField,
	Σprime::TemporaryField,
	U::Gaugefield,
	C::TemporaryField,
	Q::TemporaryField,
	ρ,
)
	Λ = similar(Σ)
	staple_eachsite!(C, U)
	calc_Qmatrices!(Q, C, U, ρ)
	calc_Λmatrices!(Λ, Σ, Q, U)
	NX, NY, NZ, NT = size(Σprime)

	@batch for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					site = SiteCoords(ix, iy, iz, it)
					for μ in 1:4
						Nμ = size(Σprime)[μ]
						siteμp = move(site, μ, 1, Nμ)
						for ν in 1:4
							if ν == μ
								continue
							end

							Nν = size(Σprime)[ν]
							siteνp = move(site, μ, 1, Nν)
							siteνn = move(site, μ, -1 ,Nν)
							siteμpνn = move(siteμp, ν, -1, Nν)
							force_sum = 
								U[ν][siteμp]    * U[μ][siteνp]'  * U[ν][site]'   * Λ[ν][site]   + 
								U[ν][siteμpνn]' * U[μ][siteνn]'  * Λ[μ][siteνn]  * U[ν][siteνn] + 
								U[ν][siteμpνn]' * Λ[ν][siteμpνn] * U[μ][siteνn]' * U[ν][siteνn] - 
								U[ν][siteμpνn]' * U[μ][siteνn]'  * Λ[ν][siteνn]  * U[ν][siteνn] - 
								Λ[ν][siteμp]    * U[ν][siteμp]   * U[μ][siteνp]' * U[ν][site]'  + 
								U[ν][siteμp]    * U[μ][siteνp]'  * Λ[μ][siteνp]  * U[ν][site]'

							Σ[μ][ix,iy,iz,it] = 
								Σprime[μ][ix,iy,iz,it]  * exp_iQ(Q[μ][ix,iy,iz,it]) +
								im * C[μ][ix,iy,iz,it]' * Λ[μ][ix,iy,iz,it]         -
								im * ρ * force_sum
						end
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
function calc_Λmatrices!(
	Λ::TemporaryField,
	Σprime::TemporaryField,
	Q::Gaugefield,
	U::Gaugefield,
)
	NX, NY, NZ, NT = size(M)

	@batch for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					for μ in 1:4
						Qn = Q[μ][ix,iy,iz,it]
						trQ2 = tr(Qn, Qn)

						if trQ2 > 1e-16
							f0, f1, f2, b10, b11, b12, b20, b21, b22 = calc_coefficients(Qn)
							UΣ = U[μ][ix,iy,iz,it] * Σprime[μ][ix,iy,iz,it]
							
							B1 = b10*I + b11*Qn + b12*Qn^2
							B2 = b20*I + b21*Qn + b22*Qn^2

							Γ = tr(UΣ, B1)*Qn  + tr(UΣ, B2)*Qn^2 +
								f1*UΣ + f2*Qn*UΣ + f2*UΣ*Qn

							Λ[μ][ix,iy,iz,it] = traceless_hermitian(Γ)
						end

					end
				end
			end
		end
	end
	return nothing
end

function calc_Qmatrices!(
	Q::TemporaryField,
	staples::TemporaryField,
	U::Gaugefield,
	ρ,
)
	NX, NY, NZ, NT = size(Q)

	@batch for it in 1:NT
		for iz in 1:NZ
			for iy in 1:NY
				for ix in 1:NX
					Ω = ρ * U[1][ix,iy,iz,it] * staples[1][ix,iy,iz,it]'
					Q[1][ix,iy,iz,it] = im * traceless_antihermitian(Ω)

					Ω = ρ * U[2][ix,iy,iz,it] * staples[2][ix,iy,iz,it]'
					Q[2][ix,iy,iz,it] = im * traceless_antihermitian(Ω)

					Ω = ρ * U[3][ix,iy,iz,it] * staples[3][ix,iy,iz,it]'
					Q[3][ix,iy,iz,it] = im * traceless_antihermitian(Ω)

					Ω = ρ * U[4][ix,iy,iz,it] * staples[4][ix,iy,iz,it]'
					Q[4][ix,iy,iz,it] = im * traceless_antihermitian(Ω)
				end
			end
		end
	end

	return nothing
end
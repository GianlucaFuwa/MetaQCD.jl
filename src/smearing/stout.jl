struct Stoutsmearing <: Abstractsmearing
	numlayers::Int64
	ρ::Float64
end

function Base.length(s::Stoutsmearing)
	return s.numlayers
end

function get_ρ(s::Stoutsmearing)
	return s.ρ
end

function apply_smearing_U(U::T, smearing::Stoutsmearing) where {T<:Gaugefield}
	numlayers = length(smearing)
	ρ = get_ρ(smearing)
	Uout_multi = Vector{Gaugefield}(undef, numlayers)
	staples_multi = Vector{Temporary_field}(undef, numlayers-1)
	Qs_multi = Vector{Temporary_field}(undef, numlayers-1)
	for i = 1:numlayers
		Uout_multi[i] = similar(U)
		if i != numlayers
			staples_multi[i] = Temporary_field(U)
			Qs_multi[i] = Temporary_field(U)
		end
	end
	calc_stout_multi!(Uout_multi, staples_multi, Qs_multi, U, ρ, numlayers)
	return Uout_multi, staples_multi, Qs_multi
end

function calc_stout_multi!(
	Uout_multi::Vector{T1}, 
	staples_multi::Vector{T2}, 
	Qs_multi::Vector{T2}, 
	Uin::T1, 
	ρ::Float64, 
	numlayers,
	) where {T1<:Gaugefield, T2<:Temporary_field}

	Utmp = similar(Uin)
	staplestmp = Temporary_field(Utmp)
	Qstmp = Temporary_field(Utmp)
	U = deepcopy(Uin)
	for i = 1:numlayers
		if i != numlayers
			apply_stout_smearing!(Utmp, U, staplestmp, Qstmp, ρ)
			Uout_multi[i] = deepcopy(Utmp)
			staples_multi[i] = deepcopy(staplestmp)
			Qs_multi[i] = deepcopy(Qstmp)
			Utmp, U = U, Utmp
		else
			apply_stout_smearing!(Uin, U, staplestmp, Qstmp, ρ)
		end
	end
	return nothing
end

function apply_stout_smearing!(Uout::T, U::T, staples::T1, Qs::T1, ρ::Float64) where {T<:Gaugefield, T1<:Temporary_field}
	NX, NY, NZ, NT = size(Uout)
	staple_eachsite!(staples, U)
	calc_Qmatrices!(Qs, staples, U, ρ)
	
	@batch for it = 1:NT
		for iz = 1:NZ
			for iy = 1:NY
				for ix = 1:NX
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

"""
Stout-Force recursion \\
See: hep-lat/0311018 by Morningstar & Peardon \\
\\
Σμ = Σμ'⋅exp(iQμ) + iCμ†⋅Λμ - i∑{...}   
"""
function layer_backprop!(Σprev::T1, Σcurrent::T1, Uprev::T2, staples::T1, Qs::T1, ρ::Float64) where {T1<:Temporary_field, T2<:Gaugefield}
	Λs = similar(Σprev)

	staple_eachsite!(Cs, Uprev)
	calc_Qmatrices!(Qs, Cs, Uprev, ρ)
	calc_Λmatrices!(Λs, Σprev, Qs, Uprev)

	NX, NY, NZ, NT = size(Σcurrent)
	@batch for it = 1:NT
		for iz = 1:NZ
			for iy = 1:NY
				for ix = 1:NX
					for μ = 1:4
						for ν = 1:4
							if ν == μ
								continue
							end
							Nμ = size(Σcurrent)[μ]
							Nν = size(Σcurrent)[ν]
							site = Site_coords(ix,iy,iz,it)
							siteμp = move(site, μ, 1, Nμ)
							siteνp = move(site, μ, 1, Nν)
							siteνn = move(site, μ, -1 ,Nν)
							siteμpνn = move(siteμp, ν, -1, Nν)
							force_sum = 
								Uprev[ν][siteμp] * Uprev[μ][siteνp]' * Uprev[ν][site]' * Λs[ν][site] + 
								Uprev[ν][siteμpνn]' * Uprev[μ][siteνn]' * Λs[μ][siteνn] * Uprev[ν][siteνn] + 
								Uprev[ν][siteμpνn]' * Λs[ν][siteμpνn] * Uprev[μ][siteνn]' * Uprev[ν][siteνn] - 
								Uprev[ν][siteμpνn]' * Uprev[μ][siteνn]' * Λs[ν][siteνn] * Uprev[ν][siteνn] - 
								Λs[ν][siteμp] * Uprev[ν][siteμp] * Uprev[μ][siteνp]' * Uprev[ν][site]' + 
								Uprev[ν][siteμp] * Uprev[μ][siteνp]' * Λ[μ][siteνp] * Uprev[ν][site]'

							Σprev[μ][ix,iy,iz,it] = 
								Σcurrent[μ][ix,iy,iz,it] * exp_iQ(Qs[μ][ix,iy,iz,it]) +
								im * staples[μ][ix,iy,iz,it]' * Λs[μ][ix,iy,iz,it] -
								im * ρ * force_sum
						end
					end
				end
			end
		end
	end
	return nothing
end

function calc_Qmatrices!(Q::T1, staples::T1, U::T2, ρ::Float64) where {T1<:Temporary_field, T2<:Gaugefield}
	NX, NY, NZ, NT = size(Q)
	@batch for it = 1:NT
		for iz = 1:NZ
			for iy = 1:NY
				for ix = 1:NX
					Ω = staples[1][ix,iy,iz,it] * ρ * U[1][ix,iy,iz,it]'
					Q[1][ix,iy,iz,it] = -im * Traceless_antihermitian(Ω)

					Ω = staples[2][ix,iy,iz,it] * ρ * U[2][ix,iy,iz,it]'
					Q[2][ix,iy,iz,it] = -im * Traceless_antihermitian(Ω)

					Ω = staples[3][ix,iy,iz,it] * ρ * U[3][ix,iy,iz,it]'
					Q[3][ix,iy,iz,it] = -im * Traceless_antihermitian(Ω)

					Ω = staples[4][ix,iy,iz,it] * ρ * U[4][ix,iy,iz,it]'
					Q[4][ix,iy,iz,it] = -im * Traceless_antihermitian(Ω)
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
function calc_Λmatrices!(Λ::T1, Σprev::T1, Q::T1, U::T2) where {T1<:Temporary_field, T2<:Gaugefield}
	NX, NY, NZ, NT = size(M)
	@batch for it = 1:NT
		for iz = 1:NZ
			for iy = 1:NY
				for ix = 1:NX
					for μ = 1:4
						Qn = Q[μ][ix,iy,iz,it]
						trQ2 = trAB(Qn)
						if trQ2 > 1e-16
							f0, f1, f2, b10, b11, b12, b20, b21, b22 = calc_coefficients_Q(Qn)
							Un = U[μ][ix,iy,iz,it]
							Σn = Σprev[μ][ix,iy,iz,it]
							UnΣn = Un * Σn
							
							B1 = b10*I + b11*Qn + b12*Q^2
							B2 = b20*I + b21*Qn + b22*Q^2

							Γ = trAB(UnΣn, B=B1)*Q +  trAB(UnΣn, B=B2)*Q^2 +
								f1*UnΣn + f2*Q*UnΣn + f2*UnΣn*Q

							Λ[μ][ix,iy,iz,it] = Traceless_hermitian(Γ)
						end
					end
				end
			end
		end
	end
	return nothing
end

function stout_recursion!(Σcurrent::T1, Uout_multi::Union{Nothing,Vector{T2}}, staples_multi::Union{Nothing,T1}, Qs_multi::Union{Nothing,T1}, smearing) where {T1<:Temporary_field, T2<:Gaugefield}
	if Uout_multi === nothing || staples_multi === nothing || Qs_multi === nothing
		return nothing
	end
	numlayers = length(smearing)
	Σprev = deepcopy(Σlast)
	for i = numlayers:-1:2
		layer_backprop!(Σprev, Σcurrent, Uout_multi[i-1], staples_multi[i-1], Qs_multi[i-1], ρ)
		Σcurrent, Σprev = Σprev, Σcurrent
	end
	return nothing
end

	

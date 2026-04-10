"""
	StoutSmearing(U::Gaugefield; numlayers=0, rho=0)
	
Struct StoutSmearing holds all fields relevant to smearing and subsequent recursion. \\
Since we never actually use the smeared fields in main, they dont have to leave this scope
"""
struct StoutSmearing{TG,TT,TC,TL} <: AbstractSmearing
    numlayers::Int64
    ρ::Float64
    Usmeared_multi::Vector{TG}
    C_multi::Vector{TT}
    Q_multi::Vector{TC}
    Λ::TL
    function StoutSmearing(U::TG; numlayers=0, rho=0) where {TG}
        @assert numlayers >= 0 && rho >= 0 "number of stout layers and ρ must be >= 0"

        if numlayers == 0 || rho == 0
            return NoSmearing()
        else
            C_multi = [Colorfield(U; no_halo=true) for _ in 1:numlayers]
            Q_multi = [Expfield(U; no_halo=true) for _ in 1:numlayers]
            Usmeared_multi = [similar(U) for _ in 1:numlayers+1]
            Λ = Colorfield(U)
            return new{TG,typeof(C_multi[1]),typeof(Q_multi[1]),typeof(Λ)}(
                numlayers, rho, Usmeared_multi, C_multi, Q_multi, Λ
            )
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", stout::StoutSmearing)
    print(io, "StoutSmearing(; numlayers = $(stout.numlayers), rho = $(stout.ρ))")
    return nothing
end

function Base.show(io::IO, stout::StoutSmearing)
    print(io, "StoutSmearing(; numlayers = $(stout.numlayers), rho = $(stout.ρ))")
    return nothing
end

@inline function Base.:(==)(s1::T, s2::T) where {T<:StoutSmearing}
    return (s1.numlayers == s2.numlayers) && (s1.ρ == s2.ρ)
end

Base.length(s::StoutSmearing) = s.numlayers
get_layer(s::StoutSmearing, i) = s.Usmeared_multi[i]

function apply_smearing!(smearing, Uin)
    numlayers = length(smearing)
    Usmeared = smearing.Usmeared_multi
    C = smearing.C_multi
    Q = smearing.Q_multi
    ρ = convert(float_type(Usmeared[1]), smearing.ρ)

    copy!(Usmeared[1], Uin)

    for i in 1:numlayers
        apply_stout_smearing!(Usmeared[i+1], C[i], Q[i], Usmeared[i], ρ)
    end

    return nothing
end

function apply_stout_smearing!(Uout::Gaugefield{B,T,M}, C, Q, U, ρ) where {B,T,M}
    itr = eachindex(Uout, C, Q, U)

    parallelfor(itr, B, Val(M), (U,), (Uout, C, Q), (Uout, C, Q, U); do_edges=Val(true)) do site, (Uout, C, Q, U)
        Base.Cartesian.@nexprs 4 μ -> (
            Qμ = calc_stout_Q_kernel!(Q, C, U, site, μ, ρ);
            Uout[μ, site] = proj_onto_SU3(cmatmul_oo(exp_iQ(Qμ), U[μ, site]))
        )
    end

    return nothing
end

function stout_backprop!(Σ′, Σ, smearing, max_level=length(smearing))
    # Variable names might be misleading---the bare force Σ⁰ will be stored in Σ′, contrary
    # to the naming convention in [hep-lat/0311018]
    Usmeared = smearing.Usmeared_multi
    C = smearing.C_multi
    Q = smearing.Q_multi
    Λ = smearing.Λ

    for i in reverse(1:max_level)
        stout_recursion!(Σ, Σ′, Usmeared[i+1], Usmeared[i], C[i], Q[i], Λ, smearing.ρ)
        copy!(Σ′, Σ)
    end

    return nothing
end

"""
Stout-Force recursion \\
See [hep-lat/0311018] by Morningstar & Peardon
"""
function stout_recursion!(Σ, Σ′, U′, U::Gaugefield{B,T,M}, C, Q, Λ, ρ) where {B,T,M}
    leftmul_dagg!(Σ′, U′)
    calc_stout_Λ!(Λ, Σ′, Q, U)
    itr = eachindex(Σ, Σ′, U′, U, C, Q, Λ)

    parallelfor(itr, B, Val(M), (U, Λ), (Σ,), (Σ, Σ′, U, C, Q, Λ); do_edges=Val(true)) do site, (Σ, Σ′, U, C, Q, Λ)
        for μ in 1:4
            stout_recursion_kernel!(Σ, Σ′, U, C, Q, Λ, site, μ, ρ)
        end
    end

    return nothing
end

function stout_recursion_kernel!(Σ, Σ′, U, C, Q, Λ, site, μ, ρ)
    Nμ = axes(Σ′, μ)
    siteμ⁺ = move(site, μ, 1, Nμ)
    force_sum = zero3(float_type(U))

    @inbounds begin
        for ν in 1:4
            if ν == μ
                continue
            end

            Nν = axes(Σ′, ν)
            siteν⁺ = move(site, ν, 1, Nν)
            siteν⁻ = move(site, ν, -1, Nν)
            siteμ⁺ν⁻ = move(siteμ⁺, ν, -1, Nν)

            # bring reused matrices up to cache (can also precalculate some products)
            # Uνsiteμ⁺ = U[ν,siteμ⁺]
            # Uμsiteμ⁺ = U[μ,siteν⁺]
            # Uνsite = U[ν,site]
            # Uνsiteμ⁺ν⁻ = U[ν,siteμ⁺ν⁻]
            # Uμsiteν⁻ = U[μ,siteν⁻]
            # Uνsiteν⁻ = U[ν,siteν⁻]

            force_sum +=
            cmatmul_oddo(U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site], Λ[ν, site]) +
            cmatmul_ddoo(U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], Λ[μ, siteν⁻], U[ν, siteν⁻]) +
            cmatmul_dodo(U[ν, siteμ⁺ν⁻], Λ[ν, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻]) -
            cmatmul_ddoo(U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], Λ[ν, siteν⁻], U[ν, siteν⁻]) -
            cmatmul_oodd(Λ[ν, siteμ⁺], U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site]) +
            cmatmul_odod(U[ν, siteμ⁺], U[μ, siteν⁺], Λ[μ, siteν⁺], U[ν, site])
        end

        link = U[μ, site]
        expiQ_mat = exp_iQ(Q[μ, site])
        Σ[μ, site] = traceless_antihermitian(
            cmatmul_ooo(link, Σ′[μ, site], expiQ_mat) +
            im * cmatmul_odo(link, C[μ, site], Λ[μ, site]) -
            im * ρ * cmatmul_oo(link, force_sum),
        )
    end

    return nothing
end

function calc_stout_Λ!(Λ, Σ′, Q::Expfield{B}, U::Gaugefield{B,T,M}) where {B,T,M}
    itr = eachindex(Λ, Σ′, Q, U)

    parallelfor(itr, B, Val(M), (), (Λ,), (Λ, Σ′, Q, U)) do site, (Λ, Σ′, Q, U)
        calc_stout_Λ_kernel!(Λ, Σ′, Q, U, site, 1)
        calc_stout_Λ_kernel!(Λ, Σ′, Q, U, site, 2)
        calc_stout_Λ_kernel!(Λ, Σ′, Q, U, site, 3)
        calc_stout_Λ_kernel!(Λ, Σ′, Q, U, site, 4)
    end

    return nothing
end

@inline function calc_stout_Λ_kernel!(Λ, Σ′, Q, U, site, μ)
    @inbounds begin
        q = Q[μ, site]
        Qₘ = get_Q(q)
        UΣ′ = cmatmul_oo(U[μ, site], Σ′[μ, site])

        B₁ = get_B₁(q)
        B₂ = get_B₂(q)

        Γ =
            multr(B₁, UΣ′) * Qₘ +
            multr(B₂, UΣ′) * cmatmul_oo(Qₘ, Qₘ) +
            q.f₁ * UΣ′ +
            q.f₂ * cmatmul_oo(Qₘ, UΣ′) +
            q.f₂ * cmatmul_oo(UΣ′, Qₘ)

        Λ[μ, site] = traceless_hermitian(Γ)
    end

    return nothing
end

@inline function calc_stout_Q_kernel!(Q, C, U, site, μ, ρ)
    Cμ = ρ * staple(WilsonGaugeAction(), U, μ, site)
    @inbounds C[μ, site] = Cμ

    @inbounds Ω = cmatmul_od(Cμ, U[μ, site])
    Qμ = exp_iQ_coeffs(-im * traceless_antihermitian(Ω))
    @inbounds Q[μ, site] = Qμ
    return Qμ
end

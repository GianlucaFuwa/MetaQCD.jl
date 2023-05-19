abstract type Integrator end

struct HMCUpdate <: AbstractUpdate
    integrator!::Integrator
    steps::Int64
    Δτ::Float64
    _P::Liefield
    _Utmp::Gaugefield
    _temps::Vector{TemporaryField}
    #smearing::Union{Nothing, StoutSmearing}
    meta_enabled::Bool

    function HMCUpdate(integrator, steps, Δτ, U; meta_enabled = false)
        integrator = getfield(AbstractUpdateModule, Symbol(integrator))
        #if !stoutsmearing
        #    numsmear = nothing
        #end
        _P = Liefield(U)
        _Utmp = similar(U)

        # temps for force, staples and fieldstrength tensors (incase we use metadynamics)
        numtemps = meta_enabled ? 6 : 2
        _temps = Vector{TemporaryField}(undef, numtemps)

        for i in 1:numtemps
            _temps[i] = TemporaryField(U)
        end

        return new(
			integrator(),
			steps,
			Δτ,
			_P,
			_Utmp,
            _temps,
			meta_enabled,
		)
    end
end

function update!(
    updatemethod::HMCUpdate,
    U::Gaugefield,
    rng,
    verbose::VerboseLevel;
    Bias = nothing,
    metro_test = true,
)
    # initialize_md!(updatemethod, U, rng)
    substitute_U!(updatemethod._Utmp, U)
    gaussian_momenta!(updatemethod._P, rng)
    
    Sg_old = calc_gauge_action(U)
    trP2_old = -calc_kinetic_energy(updatemethod._P)
    CV_old = U.CV

    updatemethod.integrator!(
        U,
        updatemethod,
        Bias,
    )

    Sg_new = calc_gauge_action(U)
    trP2_new = -calc_kinetic_energy(updatemethod._P)
    CV_new = Bias !== nothing ? top_charge(U, Bias.kind_of_cv) : 0

    ΔP2 = trP2_new - trP2_old
    ΔSg = Sg_new - Sg_old
    ΔV = Bias !== nothing ? (Bias(CV_new) - Bias(CV_old)) : 0
    ΔH = ΔP2 + ΔSg + ΔV

    println_verbose2(
        verbose,
        "ΔP2 = ", ΔP2, "\n",
        "ΔS = ", ΔSg, "\n",
        "ΔBias = ", ΔV, "\n",
        "ΔH = ", ΔH,
    )
    accept = metro_test ? rand(rng) ≤ exp(-ΔH) : true

    if accept
        U.Sg = Sg_new
        println_verbose2(verbose, "Accepted")
    else
        substitute_U!(U, updatemethod._Utmp)
        println_verbose2(verbose, "Rejected")
    end

    return accept
end

struct Leapfrog <: Integrator
end

function (lf::Leapfrog)(U::Gaugefield, method::HMCUpdate, Bias)
    updateP!(U, method, 0.5, Bias)

    for i = 1:method.steps-1
        updateU!(U, method, 1.0)
        updateP!(U, method, 1.0, Bias)
    end

    updateU!(U, method, 1.0)
    updateP!(U, method, 0.5, Bias)
    return nothing
end

struct OMF2Slow <: Integrator
    α::Float64
    β::Float64
    γ::Float64

    function OMF2_slow()
        α = 0.1931833275037836
        β = 0.5
        γ = 1.0 - 2.0 * α
        return new(α, β, γ)
    end
end

function (O2S::OMF2Slow)(U::Gaugefield, method::HMCUpdate, Bias)

    for i in 1:method.steps
        updateP!(U, method, O2S.α, Bias)
        updateU!(U, method, O2S.β)
        updateP!(U, method, O2S.γ, Bias)
        updateU!(U, method, O2S.β)
        updateP!(U, method, O2S.α, Bias)
    end

    return nothing
end

struct OMF2 <: Integrator
    α::Float64
    β::Float64
    γ::Float64

    function OMF2()
        α = 0.1931833275037836
        β = 0.5
        γ = 1.0 - 2.0 * α
        return new(α, β, γ)
    end
end

function (O2::OMF2)(U::Gaugefield, method::HMCUpdate, Bias)
    updateP!(U, method, O2.α, Bias)
    updateU!(U, method, O2.β)
    updateP!(U, method, O2.γ, Bias)
    updateU!(U, method, O2.β)

    for i in 1:method.steps-1
        updateP!(U, method, 2*O2.α, Bias)
        updateU!(U, method, O2.β)
        updateP!(U, method, O2.γ, Bias)
        updateU!(U, method, O2.β)
    end

    updateP!(U, method, O2.α, Bias)
    return nothing
end

struct OMF4Slow <: Integrator
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
    μ::Float64
    ν::Float64

    function OMF4Slow()
        α = 0.08398315262876693
        β = 0.2539785108410595
        γ = 0.6822365335719091
        δ = -0.03230286765269967
        μ = 0.5 - γ - α
        ν = 1.0 - 2.0 * δ - 2.0 * β
        return new(α, β, γ, δ, μ, ν)
    end
end

function (O4S::OMF4Slow)(U::Gaugefield, method::HMCUpdate, Bias)

    for i in 1:method.steps
        updateP!(U, method, O4S.α, Bias)
        updateU!(U, method, O4S.β)
        updateP!(U, method, O4S.γ, Bias)
        updateU!(U, method, O4S.δ)

        updateP!(U, method, O4S.μ, Bias)
        updateU!(U, method, O4S.ν)
        updateP!(U, method, O4S.μ, Bias)

        updateU!(U, method, O4S.δ)
        updateP!(U, method, O4S.γ, Bias)
        updateU!(U, method, O4S.β)
        updateP!(U, method, O4S.α, Bias)
    end

    return nothing
end

struct OMF4 <: Integrator
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
    μ::Float64
    ν::Float64

    function OMF4()
        α = 0.08398315262876693
        β = 0.2539785108410595
        γ = 0.6822365335719091
        δ = -0.03230286765269967
        μ = 0.5 - γ - α
        ν = 1.0 - 2.0 * δ - 2.0 * β
        return new(α, β, γ, δ, μ, ν)
    end
end

function (O4::OMF4)(U::Gaugefield, method::HMCUpdate, Bias)
    updateP!(U, method, O4.α, Bias)
    updateU!(U, method, O4.β)
    updateP!(U, method, O4.γ, Bias)
    updateU!(U, method, O4.δ)

    updateP!(U, method, O4.μ, Bias)
    updateU!(U, method, O4.ν)
    updateP!(U, method, O4.μ, Bias)

    updateU!(U, method, O4.δ)
    updateP!(U, method, O4.γ, Bias)
    updateU!(U, method, O4.β)

    for i in 1:method.steps-1
        updateP!(U, method, 2 * O4.α, Bias)
        updateU!(U, method, O4.β)
        updateP!(U, method, O4.γ, Bias)
        updateU!(U, method, O4.δ)

        updateP!(U, method, O4.μ, Bias)
        updateU!(U, method, O4.ν)
        updateP!(U, method, O4.μ, Bias)

        updateU!(U, method, O4.δ)
        updateP!(U, method, O4.γ, Bias)
        updateU!(U, method, O4.β)
    end

    updateP!(U, method, O4.α, Bias)
    return nothing
end

function updateU!(U::Gaugefield, method::HMCUpdate, fac)
    NX, NY, NZ, NT = size(U)
    ϵ = method.Δτ * fac
    P = method._P

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        U[μ][ix,iy,iz,it] = exp(ϵ * P[μ][ix,iy,iz,it]) * U[μ][ix,iy,iz,it]
                    end
                end
            end
        end
    end 

    return nothing
end

function updateP!(U::Gaugefield, method::HMCUpdate, fac, Bias)
    ϵ = method.Δτ * fac
    force = method._temps[1]
    staples = method._temps[2]

    if Bias !== nothing
        Fμν = method._temps[3:end]
        numlayers_cv, ρ_cv = get_smearparams_for_cv(Bias)
        smearing_cv = StoutSmearing(numlayers_cv, ρ_cv)
        Usmeared = deepcopy(U)
        Uout_multi, staples_multi, Qs_multi = calc_smearedU(Usmeared, smearing_cv)
        #calc_gauge_force!(force_toplayer, staples, Utmp)
        calc_topological_force!(force, Fμν, Usmeared, Bias.kind_of_cv)
        stout_recursion!(
            force,
            Uout_multi,
            staples_multi,
            Qs_multi,
            smearing_cv,
        )
        cv = top_charge(Usmeared, Bias.kind_of_cv)
        ϵ_bias = ϵ * return_derivative(Bias, cv)
        add_force!(method._P, force_bottomlayer, ϵ_bias)
    end

    calc_gauge_force!(force, staples, U)
    add_force!(method._P, force, ϵ)
    return nothing
end

function calc_gauge_force!(
    Σ::TemporaryField,
    staples::TemporaryField,
    U::Gaugefield,
)
    NX, NY, NZ, NT = size(U)
    β = U.β
    staple_eachsite!(staples, U)

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        tmp = U[μ][ix,iy,iz,it] * staples[μ][ix,iy,iz,it]'
                        Σ[μ][ix,iy,iz,it] = -β/6 * traceless_antihermitian(tmp)
                    end
                end
            end
        end
    end

    return nothing
end

function calc_topological_force!(
    Σ::TemporaryField,
    Fμν::Vector{TemporaryField},
    U::Gaugefield,
    kind_of_charge::String,
)
    NX, NY, NZ, NT = size(U)
    fieldstrength_eachsite!(Fμν, U, kind_of_charge)

    if kind_of_charge == "plaquette"
        ∇trFμνFρσ = ∇trFμνFρσ_clover
    elseif kind_of_charge == "clover"
        ∇trFμνFρσ = ∇trFμνFρσ_plaq
    else
        error("topological force for charge type $(kind_of_charge) not supported")
    end

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    Σ[1][ix,iy,iz,it] = 1/4π^2 * traceless_antihermitian(
                        ∇trFμνFρσ(U, Fμν, 1, 2, 3, 4, site) -
                        ∇trFμνFρσ(U, Fμν, 1, 3, 2, 4, site) +
                        ∇trFμνFρσ(U, Fμν, 1, 4, 2, 3, site)
                    )
                        
                    Σ[2][ix,iy,iz,it] = -1/4π^2 * traceless_antihermitian(
                        ∇trFμνFρσ(U, Fμν, 2, 1, 3, 4, site) +
                        ∇trFμνFρσ(U, Fμν, 2, 4, 1, 3, site) -
                        ∇trFμνFρσ(U, Fμν, 2, 3, 1, 4, site)
                    )

                    Σ[3][ix,iy,iz,it] = 1/4π^2 * traceless_antihermitian(
                        ∇trFμνFρσ(U, Fμν, 3, 4, 1, 2, site) +
                        ∇trFμνFρσ(U, Fμν, 3, 1, 2, 4, site) -
                        ∇trFμνFρσ(U, Fμν, 3, 2, 1, 4, site)
                    )

                    Σ[4][ix,iy,iz,it] = -1/4π^2 * traceless_antihermitian(
                        ∇trFμνFρσ(U, Fμν, 4, 3, 1, 2, site) -
                        ∇trFμνFρσ(U, Fμν, 4, 2, 1, 3, site) +
                        ∇trFμνFρσ(U, Fμν, 4, 1, 2, 3, site)
                    )
                end
            end
        end
    end

    return nothing
end

function add_force!(P::Liefield, force::TemporaryField, ϵ)
    NX, NY, NZ, NT = size(P)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        P[μ][ix,iy,iz,it] += ϵ * force[μ][ix,iy,iz,it] 
                    end
                end
            end
        end
    end

    return nothing
end

"""
Derivative of the F_μν ⋅ F_ρσ term for Field strength tensor given by plaquette
"""
function ∇trFμνFρσ_plaq(
    U::Gaugefield,
    Fμν::Vector{TemporaryField},
    μ,
    ν,
    ρ,
    σ,
    site::SiteCoords,
)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    siteμp = move(site, μ, 1, Nμ)
    siteνp = move(site, ν, 1, Nν)
    siteνn = move(site, ν, -1, Nν)
    siteμpνn = move(siteμp, ν, -1, Nν)

    component = 
        U[ν][siteμp] * U[μ][siteνp]' * U[ν][site]' * Fμν[ρ][σ][site] +
        U[ν][siteμpνn]' * U[μ][siteνn]' * Fμν[ρ][σ][siteνn] * U[ν][siteνn] -

    return im/2 * component
end

"""
Derivative of the F_μν ⋅ F_ρσ term for Field strength tensor given by 1x1-Clover
"""
function ∇trFμνFρσ_clover(
    U::Gaugefield,
    Fμν::Vector{TemporaryField},
    μ,
    ν,
    ρ,
    σ,
    site::SiteCoords,
)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    siteμp = move(site, μ, 1, Nμ)
    siteνp = move(site, ν, 1, Nν)
    siteνn = move(site, ν, -1, Nν)
    siteμpνp = move(siteμp, ν, 1, Nν)
    siteμpνn = move(siteμp, ν, -1, Nν)

    component = 
        U[ν][siteμp] * U[μ][siteνp]' * U[ν][site]' * Fμν[ρ][σ][site] +
        U[ν][siteμp] * U[μ][siteνp]' * Fμν[ρ][σ][siteνp] * U[ν][site]' +
        U[ν][siteμp] * Fμν[ρ][σ][siteμpνp] * U[μ][siteνp]' * U[ν][site]' +
        Fμν[ρ][σ][siteμp] * U[ν][siteμp] * U[μ][siteνp]' * U[ν][site]' -
        U[ν][siteμpνn]' * U[μ][siteνn]' * U[ν][siteνn] * Fμν[ρ][σ][site] -
        U[ν][siteμpνn]' * U[μ][siteνn]' * Fμν[ρ][σ][siteνn] * U[ν][siteνn] -
        U[ν][siteμpνn]' * Fμν[ρ][σ][siteμpνn] * U[μ][siteνn]' * U[ν][siteνn] -
        Fμν[ρ][σ][siteμp] * U[ν][siteμpνn]' * U[μ][siteνn]' * U[ν][siteνn]

    return im/8 * component
end
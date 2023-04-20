abstract type Integrator end

struct HMC_update <: AbstractUpdate
    integrator!::Integrator
    steps::Int64
    Δτ::Float64
    U::Gaugefield
    P::Liefield
    Utmp::Gaugefield
    #stoutsmearing::Bool
    #numsmear::Union{Nothing,Int64}
    meta_enabled::Bool

    function HMC_update(integrator, steps, Δτ, U, P; meta_enabled=false)
        integrator = getfield(AbstractUpdate_module, Symbol(integrator))
        #if !stoutsmearing
        #    numsmear = nothing
        #end
        Utmp = similar(U)
        return new(integrator(), steps, Δτ, U, P, Utmp, meta_enabled)
    end
end

get_momenta(hmc::HMC_update) = hmc.P
get_Δτ(hmc::HMC_update) = hmc.Δτ
get_steps(hmc::HMC_update) = hmc.steps
get_Utmp(hmc::HMC_update) = hmc.Utmp
is_meta(hmc::HMC_update) = hmc.meta_enabled

function update!(
    updatemethod::T,
    U::Gaugefield,
    rng::Xoshiro,
    verbose::Verbose_level;
    metro_test::Bool = true,
    ) where {T<:HMC_update}

    P = get_momenta(updatemethod)
    Δτ = get_Δτ(updatemethod)
    steps = get_steps(updatemethod)

    Uold = deepcopy(U)
    gaussianP!(P, rng)
    
    Sg_old = calc_GaugeAction(U)
    trP2_old = -trP2(P)

    updatemethod.integrator!(U, P, Δτ, steps)

    Sg_new = calc_GaugeAction(U)
    trP2_new = -trP2(P)

    ΔP2 = trP2_new - trP2_old
    ΔSg = Sg_new - Sg_old
    ΔH = ΔP2  + ΔSg
    println_verbose2(verbose, "ΔP2 = ", ΔP2)
    println_verbose2(verbose, "ΔS = ", ΔSg)
    println_verbose2(verbose, "ΔH = ", ΔH)
    accept = metro_test ? rand(rng)≤exp(-ΔH) : true
    if accept
        U.Sg = Sg_new
        println_verbose2(verbose, "Accepted")
    else
        substitute_U!(U, Uold)
        println_verbose2(verbose, "Rejected")
    end

    return accept
end

function update!(
    updatemethod::T,
    U::Gaugefield,
    rng::Xoshiro,
    verbose::Verbose_level,
    Bias::Bias_potential;
    metro_test::Bool=true,
    ) where {T<:HMC_update}

    P = get_momenta(updatemethod)
    Δτ = get_Δτ(updatemethod)
    steps = get_steps(updatemethod)

    Uold = deepcopy(U)
    gaussianP!(P, rng)

    Sg_old = calc_GaugeAction(U)
    trP2_old = trP2(P)
    CV_old = get_CV(U)

    updatemethod.integrator!(U, P, Δτ, steps, Bias)

    Sg_new = calc_GaugeAction(U)
    CV_new = top_charge(Uout_multi[end])

    ΔV = DeltaV(Bias, CV_old, CV_new)
    ΔH = trP2_new/2 - trP2_old/2  + Sg_new- Sg_old
    println_verbose2(verbose, "ΔV = ",ΔV)
    println_verbose2(verbose, "ΔH = ",ΔH)
    accept = metro_test ? rand(rng)≤exp(-ΔH-ΔV) : true
    if accept
        U.Sg = Sg_new 
        U.CV = CV_new
        println_verbose2(verbose, "Accepted")
    else
        substitute_U!(U, Uold)
        println_verbose2(verbose, "Rejected")
    end

    return accept
end

struct Leapfrog <: Integrator
end

function (lf::Leapfrog)(U::T, P::Liefield, Δτ, steps, Bias=nothing) where {T <: Gaugefield}
    updateP!(U, P, Δτ, 0.5, Bias)
    for i = 1:steps-1
        updateU!(U, P, Δτ, 1.0)
        updateP!(U, P, Δτ, 1.0, Bias)
    end
    updateU!(U, P, Δτ, 1.0)
    updateP!(U, P, Δτ, 0.5, Bias)

    return nothing
end

struct OMF2_slow <: Integrator
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

function (O2S::OMF2_slow)(U::T, P::Liefield, Δτ, steps, Bias=nothing) where {T <: Gaugefield}
    for i = 1:steps
        updateP!(U, P, Δτ, O2S.α, Bias)
        updateU!(U, P, Δτ, O2S.β)
        updateP!(U, P, Δτ, O2S.γ, Bias)
        updateU!(U, P, Δτ, O2S.β)
        updateP!(U, P, Δτ, O2S.α, Bias)
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

function (O2::OMF2)(U::T, P::Liefield, Δτ, steps, Bias=nothing) where {T <: Gaugefield}
    updateP!(U, P, Δτ, O2.α, Bias)
    updateU!(U, P, Δτ, O2.β)
    updateP!(U, P, Δτ, O2.γ, Bias)
    updateU!(U, P, Δτ, O2.β)
    for i = 1:steps-1
        updateP!(U, P, Δτ, 2*O2.α, Bias)
        updateU!(U, P, Δτ, O2.β)
        updateP!(U, P, Δτ, O2.γ, Bias)
        updateU!(U, P, Δτ, O2.β)
    end
    updateP!(U, P, Δτ, O2.α, Bias)

    return nothing
end

struct OMF4_slow <: Integrator
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
    μ::Float64
    ν::Float64

    function OMF4_slow()
        α = 0.08398315262876693
        β = 0.2539785108410595
        γ = 0.6822365335719091
        δ = -0.03230286765269967
        μ = 0.5 - γ - α
        ν = 1.0 - 2.0 * δ - 2.0 * β
        return new(α, β, γ, δ, μ, ν)
    end
end

function (O4S::OMF4_slow)(U::T, P::Liefield, Δτ, steps, Bias=nothing) where {T <: Gaugefield}
    for i = 1:steps
        updateP!(U, P, Δτ, O4S.α, Bias)
        updateU!(U, P, Δτ, O4S.β)
        updateP!(U, P, Δτ, O4S.γ, Bias)
        updateU!(U, P, Δτ, O4S.δ)

        updateP!(U, P, Δτ, O4S.μ, Bias)
        updateU!(U, P, Δτ, O4S.ν)
        updateP!(U, P, Δτ, O4S.μ, Bias)

        updateU!(U, P, Δτ, O4S.δ)
        updateP!(U, P, Δτ, O4S.γ, Bias)
        updateU!(U, P, Δτ, O4S.β)
        updateP!(U, P, Δτ, O4S.α, Bias)
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

function (O4::OMF4)(U::T, P::Liefield, Δτ, steps, Bias=nothing) where {T <: Gaugefield}
    updateP!(U, P, Δτ, O4.α, Bias)
    updateU!(U, P, Δτ, O4.β)
    updateP!(U, P, Δτ, O4.γ, Bias)
    updateU!(U, P, Δτ, O4.δ)

    updateP!(U, P, Δτ, O4.μ, Bias)
    updateU!(U, P, Δτ, O4.ν)
    updateP!(U, P, Δτ, O4.μ, Bias)

    updateU!(U, P, Δτ, O4.δ)
    updateP!(U, P, Δτ, O4.γ, Bias)
    updateU!(U, P, Δτ, O4.β)
    for i = 1:steps-1
        updateP!(U, P, Δτ, 2*O4.α, Bias)
        updateU!(U, P, Δτ, O4.β)
        updateP!(U, P, Δτ, O4.γ, Bias)
        updateU!(U, P, Δτ, O4.δ)

        updateP!(U, P, Δτ, O4.μ, Bias)
        updateU!(U, P, Δτ, O4.ν)
        updateP!(U, P, Δτ, O4.μ, Bias)

        updateU!(U, P, Δτ, O4.δ)
        updateP!(U, P, Δτ, O4.γ, Bias)
        updateU!(U, P, Δτ, O4.β)
    end
    updateP!(U, P, Δτ, O4.α, Bias)

    return nothing
end

function updateU!(U::Gaugefield, P::Liefield, Δτ::Float64, fac::Float64)
    NX, NY, NZ, NT = size(U)
    ϵ = Δτ * fac
    @batch for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for μ = 1:4
                        U[μ][ix,iy,iz,it] = exp(ϵ*P[μ][ix,iy,iz,it]) * U[μ][ix,iy,iz,it]
                    end
                end
            end
        end
    end 
    return nothing
end

function updateP!(U::Gaugefield, P::Liefield, Δτ::Float64, fac::Float64, Bias=nothing)
    ϵ = Δτ * fac
    force = Temporary_field(U)
    if Bias !== nothing
        numlayers, ρ = get_smearparams_for_CV(Bias)
        if numlayers !== 0 || ρ !== 0.0
            smearing = Stoutsmearing(numlayers, ρ)
            Utmp = deepcopy(U)
            Uout_multi, staples_multi, Qs_multi = calc_smearedU(Utmp, smearing)
            calc_GaugeForce_toplayer!(force, U)
            stout_recursion!(force, Uout_multi, staples_multi, Qs_multi, smearing)
            topcharge = top_charge(Utmp, get_kind_of_CV(Bias)) 
            dVdQ = ReturnDerivative(Bias, topcharge)
            ϵ *= dVdQ
        end
    else
        calc_GaugeForce_toplayer!(force, U)
    end
    add_GaugeForce!(P, force, ϵ)
    return nothing
end

function calc_GaugeForce_toplayer!(Σ::Temporary_field, U::Gaugefield)
    NX, NY, NZ, NT = size(U)
    β = get_β(U)

    @batch for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    site = Site_coords(ix,iy,iz,it)
                    for μ = 1:4
                        A = staple(U, μ, site)
                        tmp = U[μ][ix,iy,iz,it] * A'
                        Σ[μ][ix,iy,iz,it] = -β/6 * Traceless_antihermitian(tmp)
                    end
                end
            end
        end
    end
    return nothing
end

function add_GaugeForce!(P::Liefield, force::Temporary_field, ϵ::Float64)
    NX, NY, NZ, NT = size(P)
    @batch for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for μ = 1:4
                        P[μ][ix,iy,iz,it] += ϵ * force[μ][ix,iy,iz,it] 
                    end
                end
            end
        end
    end
    return nothing
end
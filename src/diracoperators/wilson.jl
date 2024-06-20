"""
    WilsonDiracOperator(::Abstractfield, mass; anti_periodic=true, r=1, csw=0)
    WilsonDiracOperator(D::WilsonDiracOperator, U::Gaugefield)

Create a free Wilson Dirac Operator with mass `mass` and Wilson parameter `r`.
If `anti_periodic` is `true` the fermion fields are anti periodic in the time direction.
If `csw ≠ 0`, a clover term is included. 
This object cannot be applied to a fermion vector, since it lacks a gauge background.
A Wilson Dirac operator with gauge background is created by applying it to a `Gaugefield`
`U` like `D_gauge = D_free(U)`

# Type Parameters:
- `B`: Backend (CPU / CUDA / ROCm)
- `T`: Floating point precision
- `TF`: Type of the `Fermionfield` used to store intermediate results when using the 
        Hermitian version of the operator
- `TG`: Type of the underlying `Gaugefield`
- `C`: Boolean declaring whether the operator is clover improved or not
"""
struct WilsonDiracOperator{B,T,C,TF,TG} <: AbstractDiracOperator
    U::TG
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    mass::Float64
    κ::Float64
    r::Float64
    csw::Float64
    anti_periodic::Bool # Only in time direction
    function WilsonDiracOperator(
        f::Abstractfield{B,T}, mass; anti_periodic=true, r=1, csw=0
    ) where {B,T}
        @assert r == 1 "Only r=1 in Wilson Dirac supported for now"
        κ = 1 / (2mass + 8)
        U = nothing
        C = csw == 0 ? false : true
        temp = Fermionfield{B,T,4}(dims(f)...)
        TG = Nothing
        TF = typeof(temp)
        return new{B,T,C,TF,TG}(U, temp, mass, κ, r, csw, anti_periodic)
    end

    function WilsonDiracOperator(
        D::WilsonDiracOperator{B,T,C,TF}, U::Gaugefield{B,T}
    ) where {B,T,C,TF}
        check_dims(U, D.temp)
        TG = typeof(U)
        return new{B,T,C,TF,TG}(U, D.temp, D.mass, D.κ, D.r, D.csw, D.anti_periodic)
    end
end

function (D::WilsonDiracOperator{B,T})(U::Gaugefield{B,T}) where {B,T}
    return WilsonDiracOperator(D, U)
end

const WilsonFermionfield{B,T,A} = Fermionfield{B,T,A,4}

struct WilsonFermionAction{Nf,C,TD,CT,RI1,RI2,RT,TX} <: AbstractFermionAction
    D::TD
    cg_temps::CT
    rhmc_info_action::RI1
    rhmc_info_md::RI2
    rhmc_temps1::RT # this holds the results of multishift cg
    rhmc_temps2::RT # this holds the basis vectors in multishift cg
    Xμν::TX
    cg_tol_action::Float64
    cg_tol_md::Float64
    cg_maxiters_action::Int64
    cg_maxiters_md::Int64
    function WilsonFermionAction(
        f,
        mass;
        anti_periodic=true,
        r=1,
        csw=0,
        Nf=2,
        rhmc_spectral_bound=(mass^2, 64.0),
        rhmc_order_action=15,
        rhmc_prec_action=42,
        rhmc_order_md=10,
        rhmc_prec_md=42,
        cg_tol_action=1e-14,
        cg_tol_md=1e-12,
        cg_maxiters_action=1000,
        cg_maxiters_md=1000,
    )
        @level1("┌ Setting Wilson Fermion Action...")
        @level1("|  MASS: $(mass)")
        @level1("|  Nf: $(Nf)")
        @level1("|  r: $(r)")
        @level1("|  CSW: $(csw)")
        @level1("|  CG TOLERANCE (Action): $(cg_tol_action)")
        @level1("|  CG TOLERANCE (MD): $(cg_tol_md)")
        @level1("|  CG MAX ITERS (Action): $(cg_maxiters_action)")
        @level1("|  CG MAX ITERS (MD): $(cg_maxiters_md)")
        D = WilsonDiracOperator(f, mass; anti_periodic=anti_periodic, r=r, csw=csw)
        TD = typeof(D)

        if Nf == 2
            rhmc_info_action = nothing
            rhmc_info_md = nothing
            rhmc_temps1 = nothing
            rhmc_temps2 = nothing
            cg_temps = ntuple(_ -> Fermionfield(f), 4)
        else
            @assert Nf == 1 "Nf should be 1 or 2 (was $Nf). If you want Nf > 2, use multiple actions"
            rhmc_lambda_low = rhmc_spectral_bound[1]
            rhmc_lambda_high = rhmc_spectral_bound[2]
            @level1("|  RHMC START SPECTRAL RANGE: [$(rhmc_lambda_low), $(rhmc_lambda_high)]")
            @level1("|  RHMC ORDER (Action): $(rhmc_order_action)")
            @level1("|  RHMC ORDER (MD): $(rhmc_order_md)")
            @level1("|  RHMC PREC (Action): $(rhmc_prec_action)")
            @level1("|  RHMC PREC (MD): $(rhmc_prec_md)")
            cg_temps = ntuple(_ -> Fermionfield(f), 2)
            power = Nf//4
            rhmc_info_action = RHMCParams(
                power;
                n=rhmc_order_action,
                precision=rhmc_prec_action,
                lambda_low=rhmc_lambda_low,
                lambda_high=rhmc_lambda_high,
            )
            power = Nf//2
            rhmc_info_md = RHMCParams(
                power;
                n=rhmc_order_action,
                precision=rhmc_prec_action,
                lambda_low=rhmc_lambda_low,
                lambda_high=rhmc_lambda_high,
            )
            n_temps = max(rhmc_order_md, rhmc_order_action)
            rhmc_temps1 = ntuple(_ -> Fermionfield(f), n_temps + 1)
            rhmc_temps2 = ntuple(_ -> Fermionfield(f), n_temps + 1)
        end

        if csw != 0
            @level1("|  CSW: $(csw) -> C = true")
            C = true
            Xμν = Tensorfield(f)
        else
            @level1("|  CSW: $(csw) -> C = false")
            C = false
            Xμν = nothing
        end

        CT = typeof(cg_temps)
        RI1 = typeof(rhmc_info_action)
        RI2 = typeof(rhmc_info_md)
        RT = typeof(rhmc_temps1)
        TX = typeof(Xμν)
        @level1("└\n")
        return new{Nf,C,TD,CT,RI1,RI2,RT,TX}(
            D,
            cg_temps,
            rhmc_info_action,
            rhmc_info_md,
            rhmc_temps1,
            rhmc_temps2,
            Xμν,
            cg_tol_action,
            cg_tol_md,
            cg_maxiters_action,
            cg_maxiters_md,
        )
    end
end

function Base.show(io::IO, ::MIME"text/plain", S::WilsonFermionAction{Nf}) where {Nf}
    print(
        io,
        "WilsonFermionAction{Nf=$Nf}(; mass=$(S.D.mass), r=$(S.D.r), csw=$(S.D.csw), " *
        "cg_tol_action=$(S.cg_tol_action), cg_tol_md=$(S.cg_tol_md), " *
        "cg_maxiters_action=$(S.cg_maxiters_action), cg_maxiters_md=$(S.cg_maxiters_md))",
    )
    return nothing
end

function Base.show(io::IO, S::WilsonFermionAction{Nf}) where {Nf}
    print(
        io,
        "WilsonFermionAction{Nf=$Nf}(; mass=$(S.D.mass), r=$(S.D.r), csw=$(S.D.csw), " *
        "cg_tol_action=$(S.cg_tol_action), cg_tol_md=$(S.cg_tol_md), " *
        "cg_maxiters_action=$(S.cg_maxiters_action), cg_maxiters_md=$(S.cg_maxiters_md))",
    )
    return nothing
end

function calc_fermion_action(
    fermion_action::WilsonFermionAction{2}, U::Gaugefield, ϕ::WilsonFermionfield
)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψ, temp1, temp2, temp3 = fermion_action.cg_temps
    cg_tol = fermion_action.cg_tol_action
    cg_maxiters = fermion_action.cg_maxiters_action

    clear!(ψ) # initial guess is zero
    solve_dirac!(ψ, DdagD, ϕ, temp1, temp2, temp3, cg_tol, cg_maxiters) # ψ = (D†D)⁻¹ϕ
    Sf = dot(ϕ, ψ)
    return real(Sf)
end

function calc_fermion_action(
    fermion_action::WilsonFermionAction{1}, U::Gaugefield, ϕ::WilsonFermionfield
)
    cg_tol = fermion_action.cg_tol_action
    cg_maxiters = fermion_action.cg_maxiters_action
    rhmc = fermion_action.rhmc_info_action
    n = get_n(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψs = fermion_action.rhmc_temps1[1:n+1]
    ps = fermion_action.rhmc_temps2[1:n+1]
    temp1, temp2 = fermion_action.cg_temps

    for v in ψs
        clear!(v)
    end

    shifts = get_β_inverse(rhmc)
    coeffs = get_α_inverse(rhmc)
    α₀ = get_α0_inverse(rhmc)
    solve_dirac_multishift!(ψs, shifts, DdagD, ϕ, temp1, temp2, ps, cg_tol, cg_maxiters)
    ψ = ψs[1]
    clear!(ψ) # D⁻¹ϕ doesn't appear in the partial fraction decomp so we can use it to sum

    axpy!(α₀, ϕ, ψ)
    for i in 1:n
        axpy!(coeffs[i], ψs[i+1], ψ)
    end

    Sf = dot(ψ, ψ)
    return real(Sf)
end

function sample_pseudofermions!(ϕ, fermion_action::WilsonFermionAction{2}, U)
    D = fermion_action.D(U)
    temp = fermion_action.cg_temps[1]
    gaussian_pseudofermions!(temp)
    mul!(ϕ, adjoint(D), temp)
    return nothing
end

function sample_pseudofermions!(ϕ, fermion_action::WilsonFermionAction{Nf}, U) where {Nf}
    cg_tol = fermion_action.cg_tol_action
    cg_maxiters = fermion_action.cg_maxiters_action
    rhmc = fermion_action.rhmc_info_action
    n = get_n(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψs = fermion_action.rhmc_temps1[1:n+1]
    ps = fermion_action.rhmc_temps2[1:n+1]
    temp1, temp2 = fermion_action.cg_temps

    for v in ψs
        clear!(v)
    end

    shifts = get_β(rhmc)
    coeffs = get_α(rhmc)
    α₀ = get_α0(rhmc)
    gaussian_pseudofermions!(ϕ) # D⁻¹ϕ doesn't appear in the partial fraction decomp so we can use it to sum
    solve_dirac_multishift!(ψs, shifts, DdagD, ϕ, temp1, temp2, ps, cg_tol, cg_maxiters)

    mul!(ϕ, α₀)
    for i in 1:n
        axpy!(coeffs[i], ψs[i+1], ϕ)
    end
    return nothing
end

function solve_dirac!(
    ψ, D::T, ϕ, temp1, temp2, temp3, temp4, temp5; tol=1e-16, maxiters=1000
) where {T<:WilsonDiracOperator}
    bicg_stab!(ψ, D, ϕ, temp1, temp2, temp3, temp4, temp5; tol=tol, maxiters=maxiters)
    return nothing
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ψ::TF, D::WilsonDiracOperator{CPU,T,C,TF,TG}, ϕ::TF
) where {T,C,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass_term = T(8 + 2 * D.mass)
    csw = D.csw
    anti = D.anti_periodic
    check_dims(ψ, ϕ, U)

    @batch for site in eachindex(ψ)
        ψ[site] = wilson_kernel(U, ϕ, site, mass_term, anti, T, Val(1))
    end

    if C
        fac = T(-csw / 2)
        @batch for site in eachindex(ψ)
            ψ[site] += clover_kernel(U, ϕ, site, fac, T)
        end
    end

    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{WilsonDiracOperator{CPU,T,C,TF,TG}}, ϕ::TF
) where {T,C,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass_term = T(8 + 2 * D.parent.mass)
    csw = D.parent.csw
    anti = D.parent.anti_periodic
    check_dims(ψ, ϕ, U)

    @batch for site in eachindex(ψ)
        ψ[site] = wilson_kernel(U, ϕ, site, mass_term, anti, T, Val(-1))
    end

    if C
        fac = T(-csw / 2)
        @batch for site in eachindex(ψ)
            ψ[site] += clover_kernel(U, ϕ, site, fac, T)
        end
    end

    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::DdaggerD{WilsonDiracOperator{CPU,T,C,TF,TG}}, ϕ::TF
) where {T,C,TF,TG}
    temp = D.parent.temp
    mul!(temp, D.parent, ϕ) # temp = Dϕ
    mul!(ψ, adjoint(D.parent), temp) # ψ = D†Dϕ
    return nothing
end

function wilson_kernel(U, ϕ, site, mass_term, anti, ::Type{T}, ::Val{dagg}) where {T,dagg}
    # dagg can be 1 or -1; if it's -1 then we swap (1 - γᵨ) with (1 + γᵨ) and vice versa
    # We have to wrap in a Val for the same reason as in the next comment
    NX, NY, NZ, NT = dims(U)
    ψₙ = mass_term * ϕ[site] # factor 1/2 is included at the end
    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    siteμ⁺ = move(site, 1, 1, NX)
    siteμ⁻ = move(site, 1, -1, NX)
    ψₙ -= cmvmul_spin_proj(U[1, site], ϕ[siteμ⁺], Val(-1dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[1, siteμ⁻], ϕ[siteμ⁻], Val(1dagg), Val(true))

    siteμ⁺ = move(site, 2, 1, NY)
    siteμ⁻ = move(site, 2, -1, NY)
    ψₙ -= cmvmul_spin_proj(U[2, site], ϕ[siteμ⁺], Val(-2dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[2, siteμ⁻], ϕ[siteμ⁻], Val(2dagg), Val(true))

    siteμ⁺ = move(site, 3, 1, NZ)
    siteμ⁻ = move(site, 3, -1, NZ)
    ψₙ -= cmvmul_spin_proj(U[3, site], ϕ[siteμ⁺], Val(-3dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[3, siteμ⁻], ϕ[siteμ⁻], Val(3dagg), Val(true))

    siteμ⁺ = move(site, 4, 1, NT)
    siteμ⁻ = move(site, 4, -1, NT)
    bc⁺ = boundary_factor(anti, site[4], 1, NT)
    bc⁻ = boundary_factor(anti, site[4], -1, NT)
    ψₙ -= cmvmul_spin_proj(U[4, site], bc⁺ * ϕ[siteμ⁺], Val(-4dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[4, siteμ⁻], bc⁻ * ϕ[siteμ⁻], Val(4dagg), Val(true))
    return T(0.5) * ψₙ
end

function clover_kernel(U, ϕ, site, fac, ::Type{T}) where {T}
    # Observed that it makes a difference whether we only make F antihermitian or traceless
    # antihermitian in the accuracy of the derivative --> TA makes it worse
    # is most severe when U is unsmeared
    Cₙₘ = zero(ϕ[site])

    C₁₂ = clover_square(U, 1, 2, site, 1)
    F₁₂ = antihermitian(C₁₂)
    Cₙₘ += cmvmul_color(F₁₂, σμν_spin_mul(ϕ[site], Val(1), Val(2)))

    C₁₃ = clover_square(U, 1, 3, site, 1)
    F₁₃ = antihermitian(C₁₃)
    Cₙₘ += cmvmul_color(F₁₃, σμν_spin_mul(ϕ[site], Val(1), Val(3)))

    C₁₄ = clover_square(U, 1, 4, site, 1)
    F₁₄ = antihermitian(C₁₄)
    Cₙₘ += cmvmul_color(F₁₄, σμν_spin_mul(ϕ[site], Val(1), Val(4)))

    C₂₃ = clover_square(U, 2, 3, site, 1)
    F₂₃ = antihermitian(C₂₃)
    Cₙₘ += cmvmul_color(F₂₃, σμν_spin_mul(ϕ[site], Val(2), Val(3)))

    C₂₄ = clover_square(U, 2, 4, site, 1)
    F₂₄ = antihermitian(C₂₄)
    Cₙₘ += cmvmul_color(F₂₄, σμν_spin_mul(ϕ[site], Val(2), Val(4)))

    C₃₄ = clover_square(U, 3, 4, site, 1)
    F₃₄ = antihermitian(C₃₄)
    Cₙₘ += cmvmul_color(F₃₄, σμν_spin_mul(ϕ[site], Val(3), Val(4)))
    return Complex{T}(fac * im / 4) * Cₙₘ
end

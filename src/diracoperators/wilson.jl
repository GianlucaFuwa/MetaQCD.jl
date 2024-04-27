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
"""
struct WilsonDiracOperator{B,T,TF,TG} <: AbstractDiracOperator
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
        @assert r === 1 "Only r=1 in Wilson Dirac supported for now"
        κ = 1 / (2mass + 8)
        U = nothing
        temp = Fermionfield(f)
        TG = Nothing
        TF = typeof(temp)
        return new{B,T,TF,TG}(U, temp, mass, κ, r, csw, anti_periodic)
    end

    function WilsonDiracOperator(
        D::WilsonDiracOperator{B,T,TF}, U::Gaugefield{B,T}
    ) where {B,T,TF}
        TG = typeof(U)
        return new{B,T,TF,TG}(U, D.temp, D.mass, D.κ, D.r, D.csw, D.anti_periodic)
    end
end

function (D::WilsonDiracOperator{B,T})(U::Gaugefield{B,T}) where {B,T}
    return WilsonDiracOperator(D, U)
end

const WilsonFermionfield{B,T,A} = Fermionfield{B,T,A,4}

struct WilsonFermionAction{Nf,TD,CT,RI,RT} <: AbstractFermionAction
    D::TD
    cg_temps::CT
    rhmc_info_action::RI
    rhmc_info_md::RI
    rhmc_temps1::RT # this holds the results of multishift cg
    rhmc_temps2::RT # this holds the basis vectors in multishift cg
    cg_tol::Float64
    cg_maxiters::Int64
    function WilsonFermionAction(
        f,
        mass;
        anti_periodic=true,
        r=1,
        csw=nothing,
        Nf=2,
        rhmc_order=10,
        rhmc_prec=42,
        cg_tol=1e-14,
        cg_maxiters=1000,
    )
        @level1("┌ Setting Wilson Fermion Action...")
        @level1("|  MASS: $(mass)")
        @level1("|  Nf: $(Nf)")
        @level1("|  r: $(r)")
        @level1("|  CSW: $(csw)")
        @level1("|  CG TOLERANCE: $(cg_tol)")
        D = WilsonDiracOperator(f, mass; anti_periodic=anti_periodic, r=r, csw=csw)
        TD = typeof(D)

        if Nf == 2
            rhmc_info_md = nothing
            rhmc_info_action = nothing
            rhmc_temps1 = nothing
            rhmc_temps2 = nothing
            cg_temps = ntuple(_ -> Fermionfield(f), 4)
        else
            @assert Nf == 1 "Nf should be 1 or 2 (was $Nf). If you want Nf > 2, use multiple actions"
            cg_temps = ntuple(_ -> Fermionfield(f), 2)
            power = Nf//2
            rhmc_info_md = RHMCParams(power; n=rhmc_order, precision=rhmc_prec)
            # rhmc_info_action = RHMCParams(power; n=rhmc_order, precision=rhmc_prec)
            rhmc_info_action = rhmc_info_md
            rhmc_temps1 = ntuple(_ -> Fermionfield(f), rhmc_order + 1)
            rhmc_temps2 = ntuple(_ -> Fermionfield(f), rhmc_order + 1)
        end

        CT = typeof(cg_temps)
        RI = typeof(rhmc_info_md)
        RT = typeof(rhmc_temps1)
        @level1("└\n")
        return new{Nf,TD,CT,RI,RT}(
            D,
            cg_temps,
            rhmc_info_md,
            rhmc_info_action,
            rhmc_temps1,
            rhmc_temps2,
            cg_tol,
            cg_maxiters,
        )
    end
end

function Base.show(io::IO, ::MIME"text/plain", S::WilsonFermionAction{Nf}) where {Nf}
    print(
        io,
        "WilsonFermionAction{Nf=$Nf}(; cg_tol=$(S.cg_tol), cg_maxiters=$(S.cg_maxiters))",
    )
    return nothing
end

function calc_fermion_action(
    fermion_action::WilsonFermionAction{2}, U::Gaugefield, ϕ::WilsonFermionfield
)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψ, temp1, temp2, temp3 = fermion_action.cg_temps
    cg_tol = fermion_action.cg_tol
    cg_maxiters = fermion_action.cg_maxiters

    clear!(ψ) # initial guess is zero
    solve_D⁻¹x!(ψ, DdagD, ϕ, temp1, temp2, temp3, cg_tol, cg_maxiters) # ψ = (D†D)⁻¹ϕ
    Sf = dot(ϕ, ψ)
    return real(Sf)
end

function calc_fermion_action(
    fermion_action::WilsonFermionAction{Nf}, U::Gaugefield, ϕ::WilsonFermionfield
) where {Nf}
    cg_tol = fermion_action.cg_tol
    cg_maxiters = fermion_action.cg_maxiters
    rhmc = fermion_action.rhmc_info_action
    n = rhmc.coeffs_inverse.n
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψs = fermion_action.rhmc_temps1
    ps = fermion_action.rhmc_temps2
    temp1, temp2 = fermion_action.cg_temps

    for v in ψs
        clear!(v)
    end

    shifts = rhmc.coeffs_inverse.β
    coeffs = rhmc.coeffs_inverse.α
    α₀ = rhmc.coeffs_inverse.α0
    solve_D⁻¹x_multishift!(ψs, shifts, DdagD, ϕ, temp1, temp2, ps, cg_tol, cg_maxiters)
    ψ = ψs[1]
    clear!(ψ) # D⁻¹ϕ doesn't appear in the partial fraction decomp so we can use it to sum

    axpy!(α₀, ϕ, ψ)
    for i in 1:n
        axpy!(coeffs[i], ψs[i+1], ψ)
    end

    Sf = dot(ϕ, ψ)
    return real(Sf)
end

function sample_pseudofermions!(ϕ, fermion_action::WilsonFermionAction{2}, U)
    D = fermion_action.D(U)
    temp = fermion_action.cg_temps[1]
    gaussian_pseudofermions!(temp)
    mul!(ϕ, D, temp)
    return nothing
end

function sample_pseudofermions!(ϕ, fermion_action::WilsonFermionAction{Nf}, U) where {Nf}
    cg_tol = fermion_action.cg_tol
    cg_maxiters = fermion_action.cg_maxiters
    rhmc = fermion_action.rhmc_info_action
    n = rhmc.coeffs.n
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψs = fermion_action.rhmc_temps1
    ps = fermion_action.rhmc_temps2
    temp1, temp2 = fermion_action.cg_temps

    for v in ψs
        clear!(v)
    end
    shifts = rhmc.coeffs.β
    coeffs = rhmc.coeffs.α
    α₀ = rhmc.coeffs.α0
    gaussian_pseudofermions!(ϕ) # D⁻¹ϕ doesn't appear in the partial fraction decomp so we can use it to sum
    solve_D⁻¹x_multishift!(ψs, shifts, DdagD, ϕ, temp1, temp2, ps, cg_tol, cg_maxiters)

    axpy!(α₀, ϕ, ϕ)
    for i in 1:n
        axpy!(coeffs[i], ψs[i+1], ϕ)
    end
    return nothing
end

function solve_D⁻¹x!(
    ψ, D::T, ϕ, temp1, temp2, temp3, temp4, temp5; tol=1e-16, maxiters=1000
) where {T<:WilsonDiracOperator}
    @assert dims(ψ) == dims(ϕ) == dims(D.U)
    bicg_stab!(ψ, D, ϕ, temp1, temp2, temp3, temp4, temp5; tol=tol, maxiters=maxiters)
    return nothing
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ψ::TF, D::WilsonDiracOperator{CPU,T,TF,TG}, ϕ::TF
) where {T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass_term = T(8 + 2 * D.mass)
    csw = D.csw
    anti = D.anti_periodic
    @assert dims(ψ) == dims(ϕ) == dims(U)

    @batch for site in eachindex(ψ)
        ψ[site] = wilson_kernel(U, ϕ, site, mass_term, anti, T)
    end

    if csw != 0
        fac = T(-csw / 2)
        @batch for site in eachindex(ψ)
            ψ[site] += clover_kernel(U, ϕ, site, fac, T)
        end
    end

    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{WilsonDiracOperator{CPU,T,TF,TG}}, ϕ::TF
) where {T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass_term = T(8 + 2 * D.parent.mass)
    csw = D.parent.csw
    anti = D.parent.anti_periodic
    @assert dims(ψ) == dims(ϕ) == dims(U)

    @batch for site in eachindex(ψ)
        ψ[site] = wilson_kernel(U, ϕ, site, mass_term, anti, T, Val(-1))
    end

    if csw != 0
        fac = T(-csw / 2)
        @batch for site in eachindex(ψ)
            ψ[site] += clover_kernel(U, ϕ, site, fac, T)
        end
    end

    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::DdaggerD{WilsonDiracOperator{CPU,T,TF,TG}}, ϕ::TF
) where {T,TF,TG}
    temp = D.parent.temp
    mul!(temp, adjoint(D.parent), ϕ) # temp = D†
    mul!(ψ, D.parent, temp) # ψ = DD†ϕ
    return nothing
end

function wilson_kernel(U, ϕ, site, mass_term, anti, T, ::Val{dagg}=Val(1)) where {dagg}
    # dagg can be 1 or -1; if it's -1 then we swap (1 - γᵨ) with (1 + γᵨ) and vice versa
    # We have to wrap in a Val for the same reason as in the next comment
    NX, NY, NZ, NT = dims(U)
    ψₙ = mass_term * ϕ[site] # factor 1/2 is included at the end
    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    siteμ⁺ = move(site, 1i32, 1i32, NX)
    siteμ⁻ = move(site, 1i32, -1i32, NX)
    ψₙ -= cmvmul_spin_proj(U[1i32, site], ϕ[siteμ⁺], Val(-1dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[1i32, siteμ⁻], ϕ[siteμ⁻], Val(1dagg), Val(true))

    siteμ⁺ = move(site, 2i32, 1i32, NY)
    siteμ⁻ = move(site, 2i32, -1i32, NY)
    ψₙ -= cmvmul_spin_proj(U[2i32, site], ϕ[siteμ⁺], Val(-2dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[2i32, siteμ⁻], ϕ[siteμ⁻], Val(2dagg), Val(true))

    siteμ⁺ = move(site, 3i32, 1i32, NZ)
    siteμ⁻ = move(site, 3i32, -1i32, NZ)
    ψₙ -= cmvmul_spin_proj(U[3i32, site], ϕ[siteμ⁺], Val(-3dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[3i32, siteμ⁻], ϕ[siteμ⁻], Val(3dagg), Val(true))

    siteμ⁺ = move(site, 4i32, 1i32, NT)
    siteμ⁻ = move(site, 4i32, -1i32, NT)
    bc⁺ = boundary_factor(anti, site[4i32], 1i32, NT)
    bc⁻ = boundary_factor(anti, site[4i32], -1i32, NT)
    ψₙ -= cmvmul_spin_proj(U[4i32, site], bc⁺ * ϕ[siteμ⁺], Val(-4dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[4i32, siteμ⁻], bc⁻ * ϕ[siteμ⁻], Val(4dagg), Val(true))
    return T(0.5) * ψₙ
end

function clover_kernel(U, ϕ, site, fac, T)
    Cₙₘ = zero(ϕ[site])

    C₁₂ = clover_square(U, 1i32, 2i32, site, 1i32)
    F₁₂ = traceless_antihermitian(C₁₂)
    Cₙₘ += cmvmul(ckron(σ₁₂(T), F₁₂), ϕ[site])

    C₁₃ = clover_square(U, 1i32, 3i32, site, 1i32)
    F₁₃ = traceless_antihermitian(C₁₃)
    Cₙₘ += cmvmul(ckron(σ₁₃(T), F₁₃), ϕ[site])

    C₁₄ = clover_square(U, 1i32, 4i32, site, 1i32)
    F₁₄ = traceless_antihermitian(C₁₄)
    Cₙₘ += cmvmul(ckron(σ₁₄(T), F₁₄), ϕ[site])

    C₂₃ = clover_square(U, 2i32, 3i32, site, 1i32)
    F₂₃ = traceless_antihermitian(C₂₃)
    Cₙₘ += cmvmul(ckron(σ₂₃(T), F₂₃), ϕ[site])

    C₂₄ = clover_square(U, 2i32, 4i32, site, 1i32)
    F₂₄ = traceless_antihermitian(C₂₄)
    Cₙₘ += cmvmul(ckron(σ₂₄(T), F₂₄), ϕ[site])

    C₃₄ = clover_square(U, 3i32, 4i32, site, 1i32)
    F₃₄ = traceless_antihermitian(C₃₄)
    Cₙₘ += cmvmul(ckron(σ₃₄(T), F₃₄), ϕ[site])
    return (fac * im * T(1 / 4)) * Cₙₘ
end

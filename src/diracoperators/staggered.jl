"""
    StaggeredDiracOperator(::Abstractfield, mass; anti_periodic=true)
    StaggeredDiracOperator(D::StaggeredDiracOperator, U::Gaugefield)

Create a free Staggered Dirac Operator with mass `mass`.
If `anti_periodic` is `true` the fermion fields are anti periodic in the time direction.
This object cannot be applied to a fermion vector, since it lacks a gauge background.
A Staggered Dirac operator with gauge background is created by applying it to a `Gaugefield`
`U` like `D_gauge = D_free(U)`

# Type Parameters:
- `B`: Backend (CPU / CUDA / ROCm)
- `T`: Floating point precision
- `TF`: Type of the `Fermionfield` used to store intermediate results when using the 
        Hermitian version of the operator
- `TG`: Type of the underlying `Gaugefield`
"""
struct StaggeredDiracOperator{B,T,TF,TG} <: AbstractDiracOperator
    U::TG
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    mass::Float64
    anti_periodic::Bool # Only in time direction
    function StaggeredDiracOperator(
        f::Abstractfield{B,T}, mass; anti_periodic=anti_periodic
    ) where {B,T}
        U = nothing
        temp = Fermionfield(f; staggered=true)
        TG = Nothing
        TF = typeof(temp)
        return new{B,T,TF,TG}(U, temp, mass, anti_periodic)
    end

    function StaggeredDiracOperator(
        D::StaggeredDiracOperator{B,T,TF}, U::Gaugefield{B,T}
    ) where {B,T,TF}
        TG = typeof(U)
        return new{B,T,TF,TG}(U, D.temp, D.mass, D.anti_periodic)
    end
end

function (D::StaggeredDiracOperator{B,T})(U::Gaugefield{B,T}) where {B,T}
    return StaggeredDiracOperator(D, U)
end

const StaggeredFermionfield{B,T,A} = Fermionfield{B,T,A,1}

struct StaggeredFermionAction{Nf,TD,CT,RI1,RI2,RT} <: AbstractFermionAction
    D::TD
    cg_temps::CT
    rhmc_info_action::RI1
    rhmc_info_md::RI2
    rhmc_temps1::RT # this holds the results of multishift cg
    rhmc_temps2::RT # this holds the basis vectors in multishift cg
    cg_tol_action::Float64
    cg_tol_md::Float64
    cg_maxiters_action::Int64
    cg_maxiters_md::Int64
    function StaggeredFermionAction(
        f,
        mass;
        anti_periodic=true,
        Nf=8,
        rhmc_order_for_md=10,
        rhmc_prec_for_md=42,
        rhmc_order_for_action=15,
        rhmc_prec_for_action=42,
        cg_tol_action=1e-14,
        cg_tol_md=1e-12,
        cg_maxiters_action=1000,
        cg_maxiters_md=1000,
    )
        @level1("┌ Setting Staggered Fermion Action...")
        @level1("|  MASS: $(mass)")
        @level1("|  Nf: $(Nf)")
        @level1("|  CG TOLERANCE (Action): $(cg_tol_action)")
        @level1("|  CG TOLERANCE (MD): $(cg_tol_md)")
        @level1("|  CG MAX ITERS (Action): $(cg_maxiters_action)")
        @level1("|  CG MAX ITERS (MD): $(cg_maxiters_md)")
        D = StaggeredDiracOperator(f, mass; anti_periodic=anti_periodic)
        TD = typeof(D)

        if Nf == 8
            rhmc_info_action = nothing
            rhmc_info_md = nothing
            rhmc_temps1 = nothing
            rhmc_temps2 = nothing
            cg_temps = ntuple(_ -> Fermionfield(f; staggered=true), 4)
        else
            @assert 8 > Nf > 0 "Nf should be between 1 and 8 (was $Nf)"
            cg_temps = ntuple(_ -> Fermionfield(f; staggered=true), 2)
            power = Nf//16
            rhmc_info_action = RHMCParams(
                power; n=rhmc_order_for_action, precision=rhmc_prec_for_action
            )
            power = Nf//8
            rhmc_info_md = RHMCParams(
                power; n=rhmc_order_for_md, precision=rhmc_prec_for_md
            )
            # rhmc_info_action = RHMCParams(power; n=rhmc_order, precision=rhmc_prec)
            n_temps = max(rhmc_order_for_md, rhmc_order_for_action)
            rhmc_temps1 = ntuple(_ -> Fermionfield(f; staggered=true), n_temps + 1)
            rhmc_temps2 = ntuple(_ -> Fermionfield(f; staggered=true), n_temps + 1)
        end

        CT = typeof(cg_temps)
        RI1 = typeof(rhmc_info_action)
        RI2 = typeof(rhmc_info_md)
        RT = typeof(rhmc_temps1)
        @level1("└\n")
        return new{Nf,TD,CT,RI1,RI2,RT}(
            D,
            cg_temps,
            rhmc_info_action,
            rhmc_info_md,
            rhmc_temps1,
            rhmc_temps2,
            cg_tol_action,
            cg_tol_md,
            cg_maxiters_action,
            cg_maxiters_md,
        )
    end
end

function Base.show(io::IO, ::MIME"text/plain", S::StaggeredFermionAction{Nf}) where {Nf}
    print(
        io,
        "StaggeredFermionAction{Nf=$Nf}(; mass=$(S.D.mass), " *
        "cg_tol_action=$(S.cg_tol_action), cg_tol_md=$(S.cg_tol_md), " *
        "cg_maxiters_action=$(S.cg_maxiters_action), cg_maxiters_md=$(S.cg_maxiters_md))",
    )
    return nothing
end

function Base.show(io::IO, S::StaggeredFermionAction{Nf}) where {Nf}
    print(
        io,
        "StaggeredFermionAction{Nf=$Nf}(; mass=$(S.D.mass), " *
        "cg_tol_action=$(S.cg_tol_action), cg_tol_md=$(S.cg_tol_md), " *
        "cg_maxiters_action=$(S.cg_maxiters_action), cg_maxiters_md=$(S.cg_maxiters_md))",
    )
    return nothing
end

function calc_fermion_action(
    fermion_action::StaggeredFermionAction{8}, U::Gaugefield, ϕ::StaggeredFermionfield
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
    fermion_action::StaggeredFermionAction{Nf}, U::Gaugefield, ϕ::StaggeredFermionfield
) where {Nf}
    cg_tol = fermion_action.cg_tol_action
    cg_maxiters = fermion_action.cg_maxiters_action
    rhmc = fermion_action.rhmc_info_action
    n = get_order(rhmc)
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

function sample_pseudofermions!(ϕ, fermion_action::StaggeredFermionAction{8}, U)
    D = fermion_action.D(U)
    temp = fermion_action.cg_temps[1]
    gaussian_pseudofermions!(temp)
    LinearAlgebra.mul!(ϕ, Daggered(D), temp)
    return nothing
end

function sample_pseudofermions!(ϕ, fermion_action::StaggeredFermionAction{Nf}, U) where {Nf}
    cg_tol = fermion_action.cg_tol_action
    cg_maxiters = fermion_action.cg_maxiters_action
    rhmc = fermion_action.rhmc_info_action
    n = get_order(rhmc)
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
    ψ, D::T, ϕ, temp1, temp2, temp3, temp4, temp5; tol=1e-14, maxiters=1000
) where {T<:StaggeredDiracOperator}
    check_dims(ψ, ϕ, D.U, temp1, temp2, temp3, temp4, temp5)
    bicg_stab!(ψ, D, ϕ, temp1, temp2, temp3, temp4, temp5; tol=tol, maxiters=maxiters)
    return nothing
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ψ::TF, D::StaggeredDiracOperator{CPU,T,TF,TG}, ϕ::TF
) where {T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass = T(D.mass)
    anti = D.anti_periodic
    check_dims(ψ, ϕ, U)

    @batch for site in eachindex(ψ)
        ψ[site] = staggered_kernel(U, ϕ, site, mass, anti, T, false)
    end

    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{StaggeredDiracOperator{CPU,T,TF,TG}}, ϕ::TF
) where {T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass = T(D.parent.mass)
    anti = D.parent.anti_periodic
    check_dims(ψ, ϕ, U)

    @batch for site in eachindex(ψ)
        ψ[site] = staggered_kernel(U, ϕ, site, mass, anti, T, true)
    end

    return nothing
end

function LinearAlgebra.mul!(
    ψ::TF, D::DdaggerD{StaggeredDiracOperator{CPU,T,TF,TG}}, ϕ::TF
) where {T,TF,TG}
    temp = D.parent.temp
    mul!(temp, D.parent, ϕ) # temp = Dϕ
    mul!(ψ, adjoint(D.parent), temp) # ψ = D†Dϕ
    return nothing
end

function staggered_kernel(U, ϕ, site, mass, anti, ::Type{T}, dagg::Bool) where {T}
    sgn = dagg ? -1 : 1
    NX, NY, NZ, NT = dims(U)
    ψₙ = 2mass * ϕ[site]
    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    siteμ⁺ = move(site, 1, 1, NX)
    siteμ⁻ = move(site, 1, -1, NX)
    η = sgn * staggered_η(Val(1), site)
    ψₙ += η * cmvmul(U[1, site], ϕ[siteμ⁺])
    ψₙ -= η * cmvmul_d(U[1, siteμ⁻], ϕ[siteμ⁻])

    siteμ⁺ = move(site, 2, 1, NY)
    siteμ⁻ = move(site, 2, -1, NY)
    η = sgn * staggered_η(Val(2), site)
    ψₙ += η * cmvmul(U[2, site], ϕ[siteμ⁺])
    ψₙ -= η * cmvmul_d(U[2, siteμ⁻], ϕ[siteμ⁻])

    siteμ⁺ = move(site, 3, 1, NZ)
    siteμ⁻ = move(site, 3, -1, NZ)
    η = sgn * staggered_η(Val(3), site)
    ψₙ += η * cmvmul(U[3, site], ϕ[siteμ⁺])
    ψₙ -= η * cmvmul_d(U[3, siteμ⁻], ϕ[siteμ⁻])

    siteμ⁺ = move(site, 4, 1, NT)
    siteμ⁻ = move(site, 4, -1, NT)
    bc⁺ = boundary_factor(anti, site[4], 1, NT)
    bc⁻ = boundary_factor(anti, site[4], -1, NT)
    η = sgn * staggered_η(Val(4), site)
    ψₙ += η * cmvmul(U[4, site], bc⁺ * ϕ[siteμ⁺])
    ψₙ -= η * cmvmul_d(U[4, siteμ⁻], bc⁻ * ϕ[siteμ⁻])
    return T(0.5) * ψₙ
end

# Use Val to reduce the amount of if-statements in the kernel
@inline staggered_η(::Val{1}, site) = 1
@inline staggered_η(::Val{2}, site) = ifelse(iseven(site[1]), 1, -1)
@inline staggered_η(::Val{3}, site) = ifelse(iseven(site[1] + site[2]), 1, -1)
@inline staggered_η(::Val{4}, site) = ifelse(iseven(site[1] + site[2] + site[3]), 1, -1)

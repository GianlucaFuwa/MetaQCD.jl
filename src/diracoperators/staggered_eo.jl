# TODO
"""
    StaggeredEOPreDiracOperator(::Abstractfield, mass; anti_periodic=true)
    StaggeredEOPreDiracOperator(
        D::Union{StaggeredDiracOperator,StaggeredEOPreDiracOperator},
        U::Gaugefield
    )

Create a free even-odd preconditioned Staggered Dirac Operator with mass `mass`.
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
struct StaggeredEOPreDiracOperator{B,T,TF,TG} <: AbstractDiracOperator
    U::TG
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    mass::Float64
    anti_periodic::Bool # Only in time direction
    function StaggeredEOPreDiracOperator(
        f::Abstractfield{B,T}, mass; anti_periodic=anti_periodic
    ) where {B,T}
        U = nothing
        temp = even_odd(Fermionfield(f; staggered=true))
        TG = Nothing
        TF = typeof(temp)
        return new{B,T,TF,TG}(U, temp, mass, anti_periodic)
    end

    function StaggeredEOPreDiracOperator(
        D::StaggeredDiracOperator{B,T,TF}, U::Gaugefield{B,T}
    ) where {B,T,TF}
        check_dims(U, D.temp)
        temp = even_odd(D.temp)
        TF_new = typeof(temp)
        TG = typeof(U)
        return new{B,T,TF_new,TG}(U, temp, D.mass, D.anti_periodic)
    end

    function StaggeredEOPreDiracOperator(
        D::StaggeredEOPreDiracOperator{B,T,TF}, U::Gaugefield{B,T}
    ) where {B,T,TF}
        check_dims(U, D.temp.parent)
        TG = typeof(U)
        return new{B,T,TF,TG}(U, D.temp, D.mass, D.anti_periodic)
    end
end

function (D::StaggeredEOPreDiracOperator{B,T})(U::Gaugefield{B,T}) where {B,T}
    return StaggeredEOPreDiracOperator(D, U)
end

const StaggeredEOPreFermionfield{B,T,A} = EvenOdd{B,T,A,1}

struct StaggeredEOPreFermionAction{Nf,TD,CT,RI1,RI2,RT} <: AbstractFermionAction
    D::TD
    cg_temps::CT
    rhmc_info_action::RI1
    rhmc_info_md::RI2
    rhmc_temps1::RT # this holds the results of multishift cg
    rhmc_temps2::RT # this holds the basis vectors in multishift cg
    cg_tol::Float64
    cg_maxiters::Int64
    function StaggeredEOPreFermionAction(
        f,
        mass;
        anti_periodic=true,
        Nf=4,
        rhmc_order_for_md=10,
        rhmc_prec_for_md=42,
        rhmc_order_for_action=15,
        rhmc_prec_for_action=42,
        cg_tol=1e-14,
        cg_maxiters=1000,
    )
        @level1("┌ Setting Even-Odd Preconditioned Staggered Fermion Action...")
        @level1("| MASS: $(mass)")
        @level1("| Nf: $(Nf)")
        D = StaggeredEOPreDiracOperator(f, mass; anti_periodic=anti_periodic)
        TD = typeof(D)

        if Nf == 4
            rhmc_info_md = nothing
            power = Nf//8
            rhmc_info_action = RHMCParams(
                power; n=rhmc_order_for_action, precision=rhmc_prec_for_action
            )
            n_temps = rhmc_order_for_action
            rhmc_temps1 = ntuple(
                _ -> even_odd(Fermionfield(f; staggered=true)), n_temps + 1
            )
            rhmc_temps2 = ntuple(
                _ -> even_odd(Fermionfield(f; staggered=true)), n_temps + 1
            )
            cg_temps = ntuple(_ -> even_odd(Fermionfield(f; staggered=true)), 4)
        else
            @assert 4 > Nf > 0 "Nf should be between 1 and 4 (was $Nf)"
            cg_temps = ntuple(_ -> even_odd(Fermionfield(f; staggered=true)), 2)
            power = Nf//4
            rhmc_info_md = RHMCParams(
                power; n=rhmc_order_for_md, precision=rhmc_prec_for_md
            )
            power = Nf//8
            rhmc_info_action = RHMCParams(
                power; n=rhmc_order_for_action, precision=rhmc_prec_for_action
            )
            n_temps = max(rhmc_order_for_md, rhmc_order_for_action)
            rhmc_temps1 = ntuple(
                _ -> even_odd(Fermionfield(f; staggered=true)), n_temps + 1
            )
            rhmc_temps2 = ntuple(
                _ -> even_odd(Fermionfield(f; staggered=true)), n_temps + 1
            )
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
            cg_tol,
            cg_maxiters,
        )
    end
end

function Base.show(
    io::IO, ::MIME"text/plain", S::StaggeredEOPreFermionAction{Nf}
) where {Nf}
    print(
        io,
        "StaggeredEOPreFermionAction{Nf=$Nf}(; mass=$(S.D.mass), " *
        "cg_tol=$(S.cg_tol), cg_maxiters=$(S.cg_maxiters))",
    )
    return nothing
end

function Base.show(io::IO, S::StaggeredEOPreFermionAction{Nf}) where {Nf}
    print(
        io,
        "StaggeredEOPreFermionAction{Nf=$Nf}(; mass=$(S.D.mass), " *
        "cg_tol=$(S.cg_tol), cg_maxiters=$(S.cg_maxiters))",
    )
    return nothing
end

function calc_fermion_action(
    fermion_action::StaggeredEOPreFermionAction{4},
    U::Gaugefield,
    ϕ_eo::StaggeredEOPreFermionfield,
)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψ_eo, temp1, temp2, temp3 = fermion_action.cg_temps
    cg_tol = fermion_action.cg_tol
    cg_maxiters = fermion_action.cg_maxiters

    clear!(ψ_eo) # initial guess is zero
    solve_dirac!(ψ_eo, DdagD, ϕ_eo, temp1, temp2, temp3, cg_tol, cg_maxiters) # ψ = (D†D)⁻¹ϕ
    Sf = dot(ϕ_eo, ψ_eo)
    return real(Sf)
end

function calc_fermion_action(
    fermion_action::StaggeredEOPreFermionAction{Nf},
    U::Gaugefield,
    ϕ_eo::StaggeredEOPreFermionfield,
) where {Nf}
    cg_tol = fermion_action.cg_tol
    cg_maxiters = fermion_action.cg_maxiters
    rhmc = fermion_action.rhmc_info_action
    n = rhmc.coeffs_inverse.n
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψs = fermion_action.rhmc_temps1[1:n+1]
    ps = fermion_action.rhmc_temps2[1:n+1]
    temp1, temp2 = fermion_action.cg_temps

    for v_eo in ψs
        clear!(v_eo)
    end

    shifts = rhmc.coeffs_inverse.β
    coeffs = rhmc.coeffs_inverse.α
    α₀ = rhmc.coeffs_inverse.α0
    solve_dirac_multishift!(ψs, shifts, DdagD, ϕ_eo, temp1, temp2, ps, cg_tol, cg_maxiters)
    ψ_eo = ψs[1]
    clear!(ψ_eo) # D⁻¹ϕ doesn't appear in the partial fraction decomp so we can use it to sum

    axpy!(α₀, ϕ_eo, ψ_eo)
    for i in 1:n
        axpy!(coeffs[i], ψs[i+1], ψ_eo)
    end

    Sf = dot(ψ_eo, ψ_eo)
    return real(Sf)
end

# function sample_pseudofermions!(ϕ_eo, fermion_action::StaggeredEOPreFermionAction{4}, U)
#     D = fermion_action.D(U)
#     temp = fermion_action.cg_temps[1]
#     gaussian_pseudofermions!(temp)
#     LinearAlgebra.mul!(ϕ_eo, D, temp)
#     return nothing
# end

function sample_pseudofermions!(
    ϕ_eo, fermion_action::StaggeredEOPreFermionAction{Nf}, U
) where {Nf}
    cg_tol = fermion_action.cg_tol
    cg_maxiters = fermion_action.cg_maxiters
    rhmc = fermion_action.rhmc_info_action
    n = rhmc.coeffs.n
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψs = fermion_action.rhmc_temps1[1:n+1]
    ps = fermion_action.rhmc_temps2[1:n+1]
    temp1, temp2 = fermion_action.cg_temps

    for v in ψs
        clear!(v)
    end
    shifts = rhmc.coeffs.β
    coeffs = rhmc.coeffs.α
    α₀ = rhmc.coeffs.α0
    gaussian_pseudofermions!(ϕ_eo) # D⁻¹ϕ doesn't appear in the partial fraction decomp so we can use it to sum
    solve_dirac_multishift!(ψs, shifts, DdagD, ϕ_eo, temp1, temp2, ps, cg_tol, cg_maxiters)

    axpy!(α₀, ϕ_eo, ϕ_eo)
    for i in 1:n
        axpy!(coeffs[i], ψs[i+1], ϕ_eo)
    end
    return nothing
end

function solve_dirac!(
    ψ_eo, D::T, ϕ_eo, temp1, temp2, temp3, temp4, temp5; tol=1e-14, maxiters=1000
) where {T<:StaggeredEOPreDiracOperator}
    bicg_stab!(ψ_eo, D, ϕ_eo, temp1, temp2, temp3, temp4, temp5; tol=tol, maxiters=maxiters)
    return nothing
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ψ_eo::TF, D::StaggeredEOPreDiracOperator{CPU,T,TF,TG}, ϕ_eo::TF
) where {T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass = T(D.mass)
    anti = D.anti_periodic

    # ψₑ = Dₒₑϕₑ
    mul_oe!(ψ_eo, U, ϕ_eo, anti, true, false)
    # ψₑ = DₑₒDₒₑϕₑ
    mul_eo!(ψ_eo, U, ψ_eo, anti, false, false)
    axpby!(mass, ϕ_eo, -1 / mass, ψ_eo) # ψₑ = Dₑₑϕₑ - DₑₒDₒₒ⁻¹Dₒₑϕₑ
    return nothing
end

function LinearAlgebra.mul!(
    ψ_eo::TF, D::Daggered{StaggeredEOPreDiracOperator{CPU,T,TF,TG}}, ϕ_eo::TF
) where {T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass = T(D.parent.mass)
    anti = D.parent.anti_periodic

    # ψₒ = Dₒₑϕₑ
    mul_oe!(ψ_eo, U, ϕ_eo, anti, true, true)
    # ψₑ = DₑₒDₒₑϕₑ
    mul_eo!(ψ_eo, U, ψ_eo, anti, false, true)
    axpby!(mass, ϕ_eo, -1 / mass, ψ_eo) # ψₑ = Dₑₑϕₑ - Dₑₒ†Dₒₒ⁻¹Dₒₑ†ϕₑ
    return nothing
end

function LinearAlgebra.mul!(
    ψ_eo::TF, D::DdaggerD{StaggeredEOPreDiracOperator{CPU,T,TF,TG}}, ϕ_eo::TF
) where {T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass = T(D.parent.mass)
    anti = D.parent.anti_periodic

    # ψₒ = Dₒₑϕₑ
    mul_oe!(ψ_eo, U, ϕ_eo, anti, true, true)
    # ψₑ = DₑₒDₒₑϕₑ
    mul_eo!(ψ_eo, U, ψ_eo, anti, false, true)
    axpby!(mass^2, ϕ_eo, -1, ψ_eo) # ψₑ = m²ϕₑ - DₑₒDₒₑϕₑ
    return nothing
end

function mul_oe!(
    ψ_eo::TF, U::Gaugefield{CPU,T}, ϕ_eo::TF, anti, into_odd, dagg::Bool; fac=1
) where {T,TF<:EvenOdd{CPU,T}}
    check_dims(ψ_eo, ϕ_eo, U)
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    fdims = dims(ψ)
    NV = ψ.NV

    @batch for site in eachindex(ψ)
        isodd(site) || continue
        _site = if into_odd
            eo_site(site, fdims..., NV)
        else
            eo_site_switch(site, fdims..., NV)
        end
        ψ[_site] = fac * staggered_eo_kernel(U, ϕ, site, anti, T, dagg)
    end
end

function mul_eo!(
    ψ_eo::TF, U::Gaugefield{CPU,T}, ϕ_eo::TF, anti, into_odd, dagg::Bool; fac=1
) where {T,TF<:EvenOdd{CPU,T}}
    check_dims(ψ_eo, ϕ_eo, U)
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    fdims = dims(ψ)
    NV = ψ.NV

    @batch for site in eachindex(ψ)
        iseven(site) || continue
        _site = if into_odd
            eo_site_switch(site, fdims..., NV)
        else
            eo_site(site, fdims..., NV)
        end
        ψ[_site] = fac * staggered_eo_kernel(U, ϕ, site, anti, T, dagg)
    end
end

function staggered_eo_kernel(U, ϕ, site, anti, ::Type{T}, dagg::Bool) where {T}
    sgn = dagg ? -1 : 1
    # sites that begin with a "_" are meant for indexing into the even-odd preconn'ed
    # fermion field 
    NX, NY, NZ, NT = dims(U)
    NV = NX * NY * NZ * NT
    ψₙ = zero(ϕ[site])
    # Cant do a for loop here because Val(μ) cannot be known at compile time and is 
    # therefore dynamically dispatched
    _siteμ⁺ = eo_site(move(site, 1, 1, NX), NX, NY, NZ, NT, NV)
    siteμ⁻ = move(site, 1, -1, NX)
    _siteμ⁻ = eo_site(siteμ⁻, NX, NY, NZ, NT, NV)
    η = sgn * staggered_η(Val(1), site)
    ψₙ += η * cmvmul(U[1, site], ϕ[_siteμ⁺])
    ψₙ -= η * cmvmul_d(U[1, siteμ⁻], ϕ[_siteμ⁻])

    _siteμ⁺ = eo_site(move(site, 2, 1, NX), NX, NY, NZ, NT, NV)
    siteμ⁻ = move(site, 2, -1, NX)
    _siteμ⁻ = eo_site(siteμ⁻, NX, NY, NZ, NT, NV)
    η = sgn * staggered_η(Val(2), site)
    ψₙ += η * cmvmul(U[2, site], ϕ[_siteμ⁺])
    ψₙ -= η * cmvmul_d(U[2, siteμ⁻], ϕ[_siteμ⁻])

    _siteμ⁺ = eo_site(move(site, 3, 1, NX), NX, NY, NZ, NT, NV)
    siteμ⁻ = move(site, 3, -1, NX)
    _siteμ⁻ = eo_site(siteμ⁻, NX, NY, NZ, NT, NV)
    η = sgn * staggered_η(Val(3), site)
    ψₙ += η * cmvmul(U[3, site], ϕ[_siteμ⁺])
    ψₙ -= η * cmvmul_d(U[3, siteμ⁻], ϕ[_siteμ⁻])

    _siteμ⁺ = eo_site(move(site, 4, 1, NX), NX, NY, NZ, NT, NV)
    siteμ⁻ = move(site, 4, -1, NX)
    _siteμ⁻ = eo_site(siteμ⁻, NX, NY, NZ, NT, NV)
    bc⁺ = boundary_factor(anti, site[4], 1, NT)
    bc⁻ = boundary_factor(anti, site[4], -1, NT)
    η = sgn * staggered_η(Val(4), site)
    ψₙ += η * cmvmul(U[4, site], bc⁺ * ϕ[_siteμ⁺])
    ψₙ -= η * cmvmul_d(U[4, siteμ⁻], bc⁻ * ϕ[_siteμ⁻])
    return T(0.5) * ψₙ
end

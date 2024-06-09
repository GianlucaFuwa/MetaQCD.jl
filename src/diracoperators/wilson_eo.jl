"""
    WilsonEOPreDiracOperator(::Abstractfield, mass; anti_periodic=true)
    WilsonEOPreDiracOperator(
        D::Union{WilsonDiracOperator,WilsonEOPreDiracOperator},
        U::Gaugefield
    )

Create a free even-odd preconditioned Wilson Dirac Operator with mass `mass`.
If `anti_periodic` is `true` the fermion fields are anti periodic in the time direction.
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
struct WilsonEOPreDiracOperator{B,T,C,TF,TG,TC,TO} <: AbstractDiracOperator
    U::TG
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    Fμν::TC # Holds the fieldstrength in case we use clover-improved fermions (C==true)
    D_ee_oo::TO
    D_oo_inv::TO
    mass::Float64
    κ::Float64
    r::Float64
    csw::Float64
    anti_periodic::Bool # Only in time direction
    function WilsonEOPreDiracOperator(
        f::Abstractfield{B,T}, mass; anti_periodic=true, r=1, csw=0
    ) where {B,T}
        @assert r === 1 "Only r=1 in Wilson Dirac supported for now"
        κ = 1 / (2mass + 8)
        U = nothing
        C = csw == 0 ? false : true
        NX, NY, NZ, NT = dims(U)
        temp = even_odd(Fermionfield(f))
        D_ee_oo = KA.zeros(B(), SMatrix{6,6,Complex{T},36}, 2, NX, NY, NZ, NT)
        D_oo_inv = KA.zeros(B(), SMatrix{6,6,Complex{T},36}, 2, NX, NY, NZ, NT÷2)

        TG = Nothing
        TF = typeof(temp)
        TO = typeof(D_ee_oo)
        return new{B,T,C,TF,TG,TO}(U, temp, D_ee_oo, D_oo_inv, mass, κ, r, csw, anti_periodic)
    end

    function WilsonEOPreDiracOperator(
        D::WilsonDiracOperator{B,T,C,TF}, U::Gaugefield{B,T}
    ) where {B,T,C,TF}
        check_dims(U, D.temp)
        NX, NY, NZ, NT = dims(U)
        temp = even_odd(D.temp)
        D_ee_oo = KA.zeros(B(), SMatrix{6,6,Complex{T},36}, 2, NX, NY, NZ, NT)
        D_oo_inv = KA.zeros(B(), SMatrix{6,6,Complex{T},36}, 2, NX, NY, NZ, NT÷2)
        calc_ee_oo!(D_ee_oo, D_oo_inv, U, D.mass)

        TF_new = typeof(temp)
        TG = typeof(U)
        TO = typeof(D_ee_oo)
        return new{B,T,C,TF_new,TG,TO}(
            U, temp, D_ee_oo, D_oo_inv, D.mass, D.κ, D.r, D.csw, D.anti_periodic
        )
    end

    function WilsonEOPreDiracOperator(
        D::WilsonEOPreDiracOperator{B,T,C,TF}, U::Gaugefield{B,T}
    ) where {B,T,C,TF}
        check_dims(U, D.temp.parent)
        calc_ee_oo!(D.D_ee_oo, D.D_oo_inv, U, D.mass)
        TG = typeof(U)
        TO = typeof(D.D_ee_oo)
        return new{B,T,C,TF,TG,TO}(
            U, D.temp, D.D_ee_oo, D.D_oo_inv, D.mass, D.κ, D.r, D.csw, D.anti_periodic
        )
    end
end

function (D::WilsonEOPreDiracOperator{B,T})(U::Gaugefield{B,T}) where {B,T}
    return WilsonEOPreDiracOperator(D, U)
end

const WilsonEOPreFermionfield{B,T,A} = EvenOdd{B,T,A,4}

struct WilsonEOPreFermionAction{Nf,TD,CT,RI,RT,TX} <: AbstractFermionAction
    D::TD
    cg_temps::CT
    rhmc_info_action::RI
    rhmc_info_md::RI
    rhmc_temps1::RT # this holds the results of multishift cg
    rhmc_temps2::RT # this holds the basis vectors in multishift cg
    Xμν::TX
    cg_tol_action::Float64
    cg_tol_md::Float64
    cg_maxiters_action::Int64
    cg_maxiters_md::Int64
    function WilsonEOPreFermionAction(
        f::Abstractfield{B,T},
        mass;
        anti_periodic=true,
        r=1,
        csw=nothing,
        Nf=2,
        rhmc_order_for_md=10,
        rhmc_prec_for_md=42,
        rhmc_order_for_action=15,
        rhmc_prec_for_action=42,
        cg_tol_action=1e-14,
        cg_tol_md=1e-12,
        cg_maxiters_action=1000,
        cg_maxiters_md=1000,
    ) where {B,T}
        @level1("┌ Setting Even-Odd Preconditioned Wilson Fermion Action...")
        @level1("|  MASS: $(mass)")
        @level1("|  Nf: $(Nf)")
        @level1("|  r: $(r)")
        @level1("|  CSW: $(csw)")
        @level1("|  CG TOLERANCE (Action): $(cg_tol_action)")
        @level1("|  CG TOLERANCE (MD): $(cg_tol_md)")
        @level1("|  CG MAX ITERS (Action): $(cg_maxiters_action)")
        @level1("|  CG MAX ITERS (MD): $(cg_maxiters_md)")
        D = WilsonEOPreDiracOperator(f, mass; anti_periodic=anti_periodic, r=r, csw=csw)
        TD = typeof(D)

        if Nf == 2
            rhmc_info_md = nothing
            rhmc_info_action = nothing
            rhmc_temps1 = nothing
            rhmc_temps2 = nothing
            cg_temps = ntuple(_ -> even_odd(Fermionfield(f)), 4)
        else
            @assert Nf == 1 "Nf should be 1 or 2 (was $Nf). If you want Nf > 2, use multiple actions"
            cg_temps = ntuple(_ -> even_odd(Fermionfield(f)), 2)
            power = Nf//2
            rhmc_info_md = RHMCParams(
                power; n=rhmc_order_for_md, precision=rhmc_prec_for_md
            )
            power = Nf//4
            rhmc_info_action = RHMCParams(
                power; n=rhmc_order_for_action, precision=rhmc_prec_for_action
            )
            n_temps = max(rhmc_order_for_md, rhmc_order_for_action)
            rhmc_temps1 = ntuple(_ -> even_odd(Fermionfield(f)), n_temps + 1)
            rhmc_temps2 = ntuple(_ -> even_odd(Fermionfield(f)), n_temps + 1)
        end

        if csw != 0
            C = true
            Xμν = Tensorfield(f)
        else
            C = false
            Xμν = nothing
        end

        CT = typeof(cg_temps)
        RI = typeof(rhmc_info_md)
        RT = typeof(rhmc_temps1)
        TX = typeof(Xμν)
        @level1("└\n")
        return new{Nf,C,TD,CT,RI,RT,TX}(
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

function Base.show(io::IO, ::MIME"text/plain", S::WilsonEOPreFermionAction{Nf}) where {Nf}
    print(
        io,
        "WilsonEOPreFermionAction{Nf=$Nf}(; mass=$(S.D.mass), r=$(S.D.r), csw=$(S.D.csw), " *
        "cg_tol_action=$(S.cg_tol_action), cg_tol_md=$(S.cg_tol_md), " *
        "cg_maxiters_action=$(S.cg_maxiters_action), cg_maxiters_md=$(S.cg_maxiters_md))",
    )
    return nothing
end

function Base.show(io::IO, S::WilsonEOPreFermionAction{Nf}) where {Nf}
    print(
        io,
        "WilsonEOPreFermionAction{Nf=$Nf}(; mass=$(S.D.mass), r=$(S.D.r), csw=$(S.D.csw), " *
        "cg_tol_action=$(S.cg_tol_action), cg_tol_md=$(S.cg_tol_md), " *
        "cg_maxiters_action=$(S.cg_maxiters_action), cg_maxiters_md=$(S.cg_maxiters_md))",
    )
    return nothing
end

function calc_fermion_action(
    fermion_action::WilsonEOPreFermionAction{4},
    U::Gaugefield,
    ϕ_eo::WilsonEOPreFermionfield,
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
    fermion_action::WilsonEOPreFermionAction{Nf},
    U::Gaugefield,
    ϕ_eo::WilsonEOPreFermionfield,
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

function solve_dirac!(
    ψ_eo, D::T, ϕ_eo, temp1, temp2, temp3, temp4, temp5; tol=1e-14, maxiters=1000
) where {T<:WilsonEOPreDiracOperator}
    check_dims(ψ_eo, ϕ_eo, D.U, temp1, temp2, temp3, temp4, temp5)
    bicg_stab!(ψ_eo, D, ϕ_eo, temp1, temp2, temp3, temp4, temp5; tol=tol, maxiters=maxiters)
    return nothing
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ψ_eo::TF, D::WilsonEOPreDiracOperator{CPU,T,TF,TG}, ϕ_eo::TF
) where {T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    check_dims(ψ_eo, ϕ_eo, U)
    anti = D.anti_periodic
    D_oo_inv = D.D_oo_inv

    mul_oe!(ψ_eo, U, ϕ_eo, anti, true, Val(false)) # ψₒ = Dₒₑϕₑ
    mul!(D_oo_inv, ψ_eo) # ψₒ = Dₒₒ⁻¹Dₒₑϕₑ
    axpy!(D_ee_oo, ϕ_eo, ψ_eo) # ψₑ = Dₑₑϕₑ - DₑₒDₒₒ⁻¹Dₒₑϕₑ
    return nothing
end

function LinearAlgebra.mul!(
    ψ_eo::TF, D::Daggered{WilsonEOPreDiracOperator{CPU,T,TF,TG}}, ϕ_eo::TF
) where {T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    check_dims(ψ_eo, ϕ_eo, U)
    anti = D.parent.anti_periodic
    D_oo_inv = D.D_oo_inv

    mul_oe!(ψ_eo, U, ϕ_eo, anti, true, Val(true)) # ψₒ = Dₒₑϕₑ
    mul!(D_oo_inv, ψ_eo) # ψₒ = Dₒₒ⁻¹Dₒₑϕₑ
    axpy!(D_ee_oo, ϕ_eo, ψ_eo) # ψₑ = Dₑₑϕₑ - DₑₒDₒₒ⁻¹Dₒₑϕₑ
    return nothing
end

function LinearAlgebra.mul!(
    ψ_eo::TF, D::DdaggerD{WilsonEOPreDiracOperator{CPU,T,TF,TG}}, ϕ_eo::TF
) where {T,TF,TG}
    temp = D.parent.temp
    mul!(temp, D.parent, ϕ_eo) # temp = Dϕ
    mul!(ψ_eo, adjoint(D.parent), temp) # ψ = D†Dϕ
    return nothing
end

function mul_oe!(
    ψ_eo::TF, U::Gaugefield{CPU,T}, ϕ_eo::TF, D_ee_oo, anti, into_odd, ::Val{dagg}; fac=1
) where {T,TF<:EvenOdd{CPU,T},dagg}
    check_dims(ψ_eo, ϕ_eo, U)
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    fdims = dims(ψ)
    NV = ψ.NV

    @batch for site in eachindex(ψ)
        isodd(site) || continue
        _site = eo_site(site, fdims..., NV)
        __site = into_odd ? _site : switch_sides(_site, fdims..., NV)
        ψ[__site] =
            fac * cmvmul(D_ee_oo[_site], wilson_eo_kernel(U, ϕ, site, anti, T, Val(dagg)))
    end
end

function mul_eo!(
    ψ_eo::TF, U::Gaugefield{CPU,T}, ϕ_eo::TF, anti, into_odd, ::Val{dagg}; fac=1
) where {T,TF<:EvenOdd{CPU,T},dagg}
    check_dims(ψ_eo, ϕ_eo, U)
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    fdims = dims(ψ)
    NV = ψ.NV

    @batch for site in eachindex(ψ)
        iseven(site) || continue
        _site = eo_site(site, fdims..., NV)
        __site = into_odd ? _site : switch_sides(_site, fdims..., NV)
        ψ[__site] =
            fac * cmvmul(D_oo_inv[_site], wilson_eo_kernel(U, ϕ, site, anti, T, Val(dagg)))
    end
end

function calc_ee_oo!(D_ee_oo, D_oo_inv, U::Gaugefield{CPU,T}, mass) where {T}
    check_dims(D_ee_oo, U)
    mass_term = Complex{T}(4 + mass)
    fdims = dims(U)
    NV = U.NV

    @batch for site in eachindex(U)
        _site = eo_site(site, fdims..., NV)
        M = @SMatrix fill(mass_term, 12, 12)

        # TODO: make σμν_spin_mul return two 6x6 matrices instead of one 12x12
        C₁₂ = clover_square(U, 1, 2, site, 1)
        F₁₂ = antihermitian(C₁₂)
        M += ckron(F₁₂, σμν_spin_mul(ϕ[_site], Val(1), Val(2)))

        C₁₃ = clover_square(U, 1, 3, site, 1)
        F₁₃ = antihermitian(C₁₃)
        M += ckron(F₁₃, σμν_spin_mul(ϕ[_site], Val(1), Val(3)))

        C₁₄ = clover_square(U, 1, 4, site, 1)
        F₁₄ = antihermitian(C₁₄)
        M += ckron(F₁₄, σμν_spin_mul(ϕ[_site], Val(1), Val(4)))

        C₂₃ = clover_square(U, 2, 3, site, 1)
        F₂₃ = antihermitian(C₂₃)
        M += ckron(F₂₃, σμν_spin_mul(ϕ[_site], Val(2), Val(3)))

        C₂₄ = clover_square(U, 2, 4, site, 1)
        F₂₄ = antihermitian(C₂₄)
        M += ckron(F₂₄, σμν_spin_mul(ϕ[_site], Val(2), Val(4)))

        C₃₄ = clover_square(U, 3, 4, site, 1)
        F₃₄ = antihermitian(C₃₄)
        M += ckron(F₃₄, σμν_spin_mul(ϕ[_site], Val(3), Val(4)))

        M1 = SMatrix{6,6,Complex{T},36}(view(M, 1:6, 1:6))
        M2 = SMatrix{6,6,Complex{T},36}(view(M, 7:12, 7:12))
        D_ee_oo[1, _site] = M1 
        D_ee_oo[2, _site] = M2 

        if isodd(site)
            D_oo_inv[1, _site] = inv(M1)
            D_oo_inv[2, _site] = inv(M2)
        end
    end
end

function wilson_eo_kernel(U, ϕ, site, anti, ::Type{T}, ::Val{dagg}) where {T,dagg}
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
    ψₙ -= cmvmul_spin_proj(U[1, site], ϕ[_siteμ⁺], Val(-1dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[1, siteμ⁻], ϕ[_siteμ⁻], Val(1dagg), Val(true))

    _siteμ⁺ = eo_site(move(site, 2, 1, NX), NX, NY, NZ, NT, NV)
    siteμ⁻ = move(site, 2, -1, NX)
    _siteμ⁻ = eo_site(siteμ⁻, NX, NY, NZ, NT, NV)
    ψₙ -= cmvmul_spin_proj(U[2, site], ϕ[_siteμ⁺], Val(-2dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[2, siteμ⁻], ϕ[_siteμ⁻], Val(2dagg), Val(true))

    _siteμ⁺ = eo_site(move(site, 3, 1, NX), NX, NY, NZ, NT, NV)
    siteμ⁻ = move(site, 3, -1, NX)
    _siteμ⁻ = eo_site(siteμ⁻, NX, NY, NZ, NT, NV)
    ψₙ -= cmvmul_spin_proj(U[3, site], ϕ[_siteμ⁺], Val(-3dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[3, siteμ⁻], ϕ[_siteμ⁻], Val(3dagg), Val(true))

    _siteμ⁺ = eo_site(move(site, 4, 1, NX), NX, NY, NZ, NT, NV)
    siteμ⁻ = move(site, 4, -1, NX)
    _siteμ⁻ = eo_site(siteμ⁻, NX, NY, NZ, NT, NV)
    bc⁺ = boundary_factor(anti, site[4], 1, NT)
    bc⁻ = boundary_factor(anti, site[4], -1, NT)
    ψₙ -= cmvmul_spin_proj(U[4, site], bc⁺ * ϕ[_siteμ⁺], Val(-4dagg), Val(false))
    ψₙ -= cmvmul_spin_proj(U[4, siteμ⁻], bc⁻ * ϕ[_siteμ⁻], Val(4dagg), Val(true))
    return T(0.5) * ψₙ
end

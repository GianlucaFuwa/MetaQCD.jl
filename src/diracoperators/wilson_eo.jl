"""
    WilsonEOPreDiracOperator(::AbstractField, mass; bc_str="antiperiodic")
    WilsonEOPreDiracOperator(
        D::Union{WilsonDiracOperator,WilsonEOPreDiracOperator},
        U::Gaugefield
    )

Create a free even-odd preconditioned Wilson Dirac Operator with mass `mass`.

`bc_str` can either be `"periodic"` or `"antiperiodic"` and specifies the boundary
condition in the time direction.

If `csw ≠ 0`, a clover term is included. 

This object cannot be applied to a fermion vector, since it lacks a gauge background.
A Wilson Dirac operator with gauge background is created by applying it to a `Gaugefield`
`U` like `D_gauge = D(U)`

# Type Parameters:
- `B`: Backend (CPU / CUDA / ROCm)
- `T`: Floating point precision
- `TF`: Type of the `Spinorfield` used to store intermediate results when using the 
        Hermitian version of the operator
- `TG`: Type of the underlying `Gaugefield`
- `C`: Boolean declaring whether the operator is clover improved or not
- `BC`: Boundary Condition in time direction
"""
struct WilsonEOPreDiracOperator{B,T,C,TF,TG,TX,TO,BC} <: AbstractDiracOperator
    U::TG
    Fμν::TX
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    D_diag::TO
    D_oo_inv::TO
    mass::Float64
    κ::Float64
    r::Float64
    csw::Float64
    boundary_condition::BC # Only in time direction
    function WilsonEOPreDiracOperator(
        f::AbstractField{B,T}, mass; bc_str="antiperiodic", r=1, csw=0, kwargs...
    ) where {B,T}
        @assert r == 1 "Only r=1 in Wilson Dirac supported for now"
        κ = 1 / (2mass + 8)
        U = nothing
        C = csw == 0 ? false : true
        Fμν = C ? Tensorfield(f) : nothing
        temp = even_odd(Spinorfield(f))
        D_diag = Paulifield(temp, csw)
        D_oo_inv = Paulifield(temp, csw)
        boundary_condition = create_bc(bc_str, f.topology)

        TG = Nothing
        TX = typeof(Fμν)
        TF = typeof(temp)
        TO = typeof(D_diag)
        BC = typeof(boundary_condition)
        return new{B,T,C,TF,TG,TX,TO,BC}(
            U, Fμν, temp, D_diag, D_oo_inv, mass, κ, r, csw, boundary_condition
        )
    end

    function WilsonEOPreDiracOperator(
        D::WilsonDiracOperator{B,T,C,TF}, U::Gaugefield{B,T}
    ) where {B,T,C,TF}
        check_dims(U, D.temp)
        mass = D.mass
        csw = D.csw
        temp = even_odd(D.temp)
        D_diag = Paulifield(temp, csw)
        D_oo_inv = Paulifield(temp, csw)

        Fμν = C ? Tensorfield(U) : nothing
        calc_diag!(D_diag, D_oo_inv, Fμν, U, mass)

        TF_new = typeof(temp)
        TX = typeof(Fμν)
        TG = typeof(U)
        TO = typeof(D_diag)
        BC = typeof(D.boundary_condition)
        return new{B,T,C,TF_new,TG,TX,TO,BC}(
            U, Fμν, temp, D_diag, D_oo_inv, mass, D.κ, D.r, csw, D.boundary_condition
        )
    end

    function WilsonEOPreDiracOperator(
        D::WilsonEOPreDiracOperator{B,T,C,TF,TG,TX,TO}, U::Gaugefield{B,T}
    ) where {B,T,C,TF,TG,TX,TO}
        check_dims(U, D.temp.parent)
        mass = D.mass
        csw = D.csw
        Fμν = D.Fμν
        temp = D.temp
        D_diag = D.D_diag
        D_oo_inv = D.D_oo_inv

        calc_diag!(D_diag, D_oo_inv, Fμν, U, mass)
        BC = typeof(D.boundary_condition)
        return new{B,T,C,TF,typeof(U),TX,TO,BC}(
            U, Fμν, temp, D_diag, D_oo_inv, mass, D.κ, D.r, csw, D.boundary_condition
        )
    end
end

function (D::WilsonEOPreDiracOperator{B,T})(U::Gaugefield{B,T}) where {B,T}
    return WilsonEOPreDiracOperator(D, U)
end

@inline has_clover_term(::WilsonEOPreDiracOperator{B,T,C}) where {B,T,C} = C
@inline has_clover_term(::Daggered{W}) where {B,T,C,W<:WilsonEOPreDiracOperator{B,T,C}} = C
@inline has_clover_term(::DdaggerD{W}) where {B,T,C,W<:WilsonEOPreDiracOperator{B,T,C}} = C

struct WilsonEOPreFermionAction{Nf,C,TD,CT,TX,RI1,RI2,RT} <: AbstractFermionAction{Nf}
    D::TD
    cg_temps::CT
    Xμν::TX
    rhmc_info_action::RI1
    rhmc_info_md::RI2
    rhmc_temps1::RT # this holds the results of multishift cg
    rhmc_temps2::RT # this holds the basis vectors in multishift cg
    cg_tol_action::Float64
    cg_tol_md::Float64
    cg_maxiters_action::Int64
    cg_maxiters_md::Int64
    function WilsonEOPreFermionAction(
        f::AbstractField{B,T},
        mass;
        bc_str="antiperiodic",
        r=1,
        csw=0,
        Nf=2,
        rhmc_order_md=10,
        rhmc_prec_md=42,
        rhmc_order_action=15,
        rhmc_prec_action=42,
        cg_tol_action=1e-14,
        cg_tol_md=1e-12,
        cg_maxiters_action=1000,
        cg_maxiters_md=1000,
        kwargs...,
    ) where {B,T}
        D = WilsonEOPreDiracOperator(f, mass; bc_str=bc_str, r=r, csw=csw)
        TD = typeof(D)

        if Nf == 2
            rhmc_info_md = nothing
            rhmc_info_action = nothing
            rhmc_temps1 = nothing
            rhmc_temps2 = nothing
            cg_temps = ntuple(_ -> even_odd(Spinorfield(f)), 4)
        else
            @assert Nf == 1 """
            Nf should be 1 or 2 (was $Nf). If you want Nf > 2, use multiple actions
            """
            cg_temps = ntuple(_ -> even_odd(Spinorfield(f)), 2)
            power = Nf//4
            rhmc_info_action = RHMCParams(
                power; n=rhmc_order_action, precision=rhmc_prec_action
            )
            power = Nf//2
            rhmc_info_md = RHMCParams(
                power; n=rhmc_order_md, precision=rhmc_prec_md
            )
            n_temps = max(rhmc_order_md, rhmc_order_action)
            rhmc_temps1 = ntuple(_ -> even_odd(Spinorfield(f)), n_temps + 1)
            rhmc_temps2 = ntuple(_ -> even_odd(Spinorfield(f)), n_temps + 1)
        end

        C = csw != 0 ? true : false
        Xμν = C ? Tensorfield(f) : nothing
        CT = typeof(cg_temps)
        TX = typeof(Xμν)
        RI1 = typeof(rhmc_info_action)
        RI2 = typeof(rhmc_info_md)
        RT = typeof(rhmc_temps1)
        return new{Nf,C,TD,CT,TX,RI1,RI2,RT}(
            D,
            cg_temps,
            Xμν,
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

function calc_fermion_action(
    fermion_action::WilsonEOPreFermionAction{2},
    U::Gaugefield,
    ϕ_eo::WilsonEOPreSpinorfield,
)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψ_eo, temp1, temp2, temp3 = fermion_action.cg_temps
    cg_tol = fermion_action.cg_tol_action
    cg_maxiters = fermion_action.cg_maxiters_action

    # clear!(ψ_eo) # initial guess is zero
    # solve_dirac!(ψ_eo, DdagD, ϕ_eo, temp1, temp2, temp3, cg_tol, cg_maxiters) # ψ = (D†D)⁻¹ϕ
    Sf = #= dot(ϕ_eo, ψ_eo) =# - 2trlog(D.D_diag, D.mass)
    return distributed_reduce(real(Sf), +, U)
end

function calc_fermion_action(
    fermion_action::WilsonEOPreFermionAction{1},
    U::Gaugefield,
    ϕ_eo::WilsonEOPreSpinorfield,
)
    error("Not implemented yet")
    # TODO:
    # return distributed_reduce(real(Sf), +, U)
end

function sample_pseudofermions!(ϕ, fermion_action::WilsonEOPreFermionAction{2}, U)
    D = fermion_action.D(U)
    temp = fermion_action.cg_temps[1]
    gaussian_pseudofermions!(temp)
    mul!(ϕ, adjoint(D), temp)
    return nothing
end

function sample_pseudofermions!(ϕ, fermion_action::WilsonEOPreFermionAction{1}, U)
    error("Not implemented yet")
    # TODO
    return nothing
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
    ψ_eo::TF, D::WilsonEOPreDiracOperator{CPU,T,C,TF,TG,TX,TO}, ϕ_eo::TF
) where {T,C,TF,TG,TX,TO}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    check_dims(ψ_eo, ϕ_eo, U)
    bc = D.boundary_condition
    D_oo_inv = D.D_oo_inv
    D_diag = D.D_diag

    mul_oe!(ψ_eo, U, ϕ_eo, bc, true, Val(1)) # ψₒ = Dₒₑϕₑ
    mul_oo_inv!(ψ_eo, D_oo_inv) # ψₒ = Dₒₒ⁻¹Dₒₑϕₑ
    mul_eo!(ψ_eo, U, ψ_eo, bc, false, Val(1)) # ψₑ = DₑₒDₒₒ⁻¹Dₒₑϕₑ
    axmy!(D_diag, ϕ_eo, ψ_eo) # ψₑ = Dₑₑϕₑ - DₑₒDₒₒ⁻¹Dₒₑϕₑ
    return nothing
end

function LinearAlgebra.mul!(
    ψ_eo::TF, D::Daggered{WilsonEOPreDiracOperator{CPU,T,C,TF,TG,TX,TO,BC}}, ϕ_eo::TF
) where {T,C,TF,TG,TX,TO,BC}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    check_dims(ψ_eo, ϕ_eo, U)
    bc = D.parent.boundary_condition
    D_oo_inv = D.parent.D_oo_inv
    D_diag = D.parent.D_diag

    mul_oe!(ψ_eo, U, ϕ_eo, bc, true, Val(-1)) # ψₒ = Dₑₒ†ϕₑ
    mul_oo_inv!(ψ_eo, D_oo_inv) # ψₒ = Dₒₒ⁻¹Dₒₑϕₑ
    mul_eo!(ψ_eo, U, ψ_eo, bc, false, Val(-1)) # ψₑ = Dₒₑ†Dₒₒ⁻¹Dₑₒ†ϕₑ
    axmy!(D_diag, ϕ_eo, ψ_eo) # ψₑ = Dₑₑϕₑ - DₑₒDₒₒ⁻¹Dₒₑϕₑ
    return nothing
end

function LinearAlgebra.mul!(
    ψ_eo::TF, D::DdaggerD{WilsonEOPreDiracOperator{CPU,T,C,TF,TG,TX,TO,BC}}, ϕ_eo::TF
) where {T,C,TF,TG,TX,TO,BC}
    temp = D.parent.temp
    mul!(temp, D.parent, ϕ_eo) # temp = Dϕ
    mul!(ψ_eo, adjoint(D.parent), temp) # ψ = D†Dϕ
    return nothing
end

function mul_oe!(
    ψ_eo::TF, U::Gaugefield{CPU,T}, ϕ_eo::TF, bc, into_odd, ::Val{dagg}; fac=1
) where {T,TF<:WilsonEOPreSpinorfield{CPU,T},dagg}
    check_dims(ψ_eo, ϕ_eo, U)
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    fdims = dims(ψ)
    NV = ψ.NV

    #= @batch  =#for site in eachindex(ψ)
        isodd(site) || continue
        _site = if into_odd
            eo_site(site, fdims..., NV)
        else
            eo_site_switch(site, fdims..., NV)
        end
        ψ[_site] = fac * wilson_eo_kernel(U, ϕ, site, bc, T, Val(dagg))
    end

    update_halo!(ψ_eo)
    return nothing
end

function mul_eo!(
    ψ_eo::TF, U::Gaugefield{CPU,T}, ϕ_eo::TF, bc, into_odd, ::Val{dagg}; fac=1
) where {T,TF<:WilsonEOPreSpinorfield{CPU,T},dagg}
    check_dims(ψ_eo, ϕ_eo, U)
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    fdims = dims(ψ)
    NV = ψ.NV

    #= @batch  =#for site in eachindex(ψ)
        iseven(site) || continue
        _site = if into_odd
            eo_site_switch(site, fdims..., NV)
        else
            eo_site(site, fdims..., NV)
        end
        ψ[_site] = fac * wilson_eo_kernel(U, ϕ, site, bc, T, Val(dagg))
    end

    update_halo!(ψ_eo)
    return nothing
end

function wilson_eo_kernel(U, ϕ, site, bc, ::Type{T}, ::Val{dagg}) where {T,dagg}
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
    ψₙ += cmvmul_spin_proj(U[1, site], ϕ[_siteμ⁺], Val(-1dagg), Val(false))
    ψₙ += cmvmul_spin_proj(U[1, siteμ⁻], ϕ[_siteμ⁻], Val(1dagg), Val(true))

    _siteμ⁺ = eo_site(move(site, 2, 1, NY), NX, NY, NZ, NT, NV)
    siteμ⁻ = move(site, 2, -1, NY)
    _siteμ⁻ = eo_site(siteμ⁻, NX, NY, NZ, NT, NV)
    ψₙ += cmvmul_spin_proj(U[2, site], ϕ[_siteμ⁺], Val(-2dagg), Val(false))
    ψₙ += cmvmul_spin_proj(U[2, siteμ⁻], ϕ[_siteμ⁻], Val(2dagg), Val(true))

    _siteμ⁺ = eo_site(move(site, 3, 1, NZ), NX, NY, NZ, NT, NV)
    siteμ⁻ = move(site, 3, -1, NZ)
    _siteμ⁻ = eo_site(siteμ⁻, NX, NY, NZ, NT, NV)
    ψₙ += cmvmul_spin_proj(U[3, site], ϕ[_siteμ⁺], Val(-3dagg), Val(false))
    ψₙ += cmvmul_spin_proj(U[3, siteμ⁻], ϕ[_siteμ⁻], Val(3dagg), Val(true))

    _siteμ⁺ = eo_site(move(site, 4, 1, NT), NX, NY, NZ, NT, NV)
    siteμ⁻ = move(site, 4, -1, NT)
    _siteμ⁻ = eo_site(siteμ⁻, NX, NY, NZ, NT, NV)
    ψₙ += cmvmul_spin_proj(
        U[4, site], apply_bc(ϕ[_siteμ⁺], bc, site, Val(1), NT), Val(-4dagg), Val(false)
    )
    ψₙ += cmvmul_spin_proj(
        U[4, siteμ⁻], apply_bc(ϕ[_siteμ⁻], bc, site, Val(-1), NT), Val(4dagg), Val(true)
    )
    return T(0.5) * ψₙ
end

function calc_diag!(
    D_diag::TW, D_oo_inv::TW, ::Nothing, U::Gaugefield{CPU,T}, mass
) where {T,M,TW<:Paulifield{CPU,T,M,false}}
    check_dims(D_diag, D_oo_inv, U)
    mass_term = Complex{T}(4 + mass)
    fdims = dims(U)
    NV = U.NV

    #= @batch  =#for site in eachindex(U)
        _site = eo_site(site, fdims..., NV)
        A = SMatrix{6,6,Complex{T},36}(mass_term * I)
        D_diag[site] = PauliMatrix(A, A)

        if isodd(site)
            o_site = switch_sides(_site, fdims..., NV)
            A_inv = SMatrix{6,6,Complex{T},36}(1/mass_term * I)
            D_oo_inv[o_site] = PauliMatrix(A_inv, A_inv)
        end
    end
end

function calc_diag!(
    D_diag::TW, D_oo_inv::TW, Fμν, U::Gaugefield{CPU,T}, mass
) where {T,MP,TW<:Paulifield{CPU,T,MP,true}} # With clover term
    check_dims(D_diag, D_oo_inv, U)
    mass_term = Complex{T}(4 + mass)
    fdims = dims(U)
    NV = U.NV
    fac = Complex{T}(-D_diag.csw / 2)

    fieldstrength_eachsite!(Clover(), Fμν, U)

    #= @batch  =#for site in eachindex(U)
        _site = eo_site(site, fdims..., NV)
        M = SMatrix{6,6,Complex{T},36}(mass_term * I)
        i = SVector((1, 2))
        j = SVector((3, 4))

        F₁₂ = Fμν[1, 2, site]
        σ = σ12(T)
        A₊ = ckron(σ[i, i], F₁₂)
        A₋ = ckron(σ[j, j], F₁₂)

        F₁₃ = Fμν[1, 3, site]
        σ = σ13(T)
        A₊ += ckron(σ[i, i], F₁₃)
        A₋ += ckron(σ[j, j], F₁₃)

        F₁₄ = Fμν[1, 4, site]
        σ = σ14(T)
        A₊ += ckron(σ[i, i], F₁₄)
        A₋ += ckron(σ[j, j], F₁₄)

        F₂₃ = Fμν[2, 3, site]
        σ = σ23(T)
        A₊ += ckron(σ[i, i], F₂₃)
        A₋ += ckron(σ[j, j], F₂₃)

        F₂₄ = Fμν[2, 4, site]
        σ = σ24(T)
        A₊ += ckron(σ[i, i], F₂₄)
        A₋ += ckron(σ[j, j], F₂₄)

        F₃₄ = Fμν[3, 4, site]
        σ = σ34(T)
        A₊ += ckron(σ[i, i], F₃₄)
        A₋ += ckron(σ[j, j], F₃₄)

        A₊ = fac * A₊ + M
        A₋ = fac * A₋ + M
        D_diag[site] = PauliMatrix(A₊, A₋) # XXX:

        # if isodd(site)
            # o_site = switch_sides(_site, fdims..., NV)
            D_oo_inv[site] = PauliMatrix(cinv(A₊), cinv(A₋)) # XXX:
        # end
    end
end

function mul_oo_inv!(
    ϕ_eo::WilsonEOPreSpinorfield{CPU,T}, D_diag::Paulifield{CPU,T}
) where {T}
    check_dims(ϕ_eo, D_diag)
    ϕ = ϕ_eo.parent
    fdims = dims(ϕ)
    NV = ϕ.NV

    #= @batch  =#for _site in eachindex(true, ϕ)
        o_site = switch_sides(_site, fdims..., NV)
        ϕ[o_site] = cmvmul_block(D_diag[_site], ϕ[o_site])
    end

    return nothing
end

function axmy!(
    D_diag::Paulifield{CPU,T}, ψ_eo::TF, ϕ_eo::TF
) where {T,TF<:SpinorfieldEO{CPU,T}} # even on even is the default
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    even = true

    #= @batch  =#for _site in eachindex(even, ϕ)
        ϕ[_site] = cmvmul_block(D_diag[_site], ψ[_site]) - ϕ[_site]
    end

    return nothing
end

function trlog(D_diag::Paulifield{CPU,T,M,false}, mass) where {T,M} # Without clover term
    NC = num_colors(D_diag)
    mass_term = Float64(4 + mass)
    logd = 4NC * log(mass_term)
    return D_diag.NV÷2 * logd
end

function trlog(D_diag::Paulifield{CPU,T,M,true}, ::Any) where {T,M} # With clover term
    d = 0.0

    @batch reduction=(+, d) for site in eachindex(D_diag) # XXX:
        p = D_diag[site] # XXX:
        d += log(real(det(p.upper)) * real(det(p.lower)))
    end

    return d
end

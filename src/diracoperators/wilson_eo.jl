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
struct WilsonEOPreDiracOperator{B,T,C,TF,TG,TX,TO} <: AbstractDiracOperator
    U::TG
    Xμν::TX
    temp::TF # temp for storage of intermediate result for DdaggerD operator
    D_diag::TO
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
        Xμν = C ? Tensorfield(U) : nothing
        temp = even_odd(Fermionfield{B,T,4}(dims(f)...))
        D_diag = WilsonEODiagonal(temp, mass, csw)
        D_oo_inv = WilsonEODiagonal(temp, mass, csw; inverse=true)

        TG = Nothing
        TX = typeof(Xμν)
        TF = typeof(temp)
        TO = typeof(D_diag)
        return new{B,T,C,TF,TG,TX,TO}(
            U, Xμν, temp, D_diag, D_oo_inv, mass, κ, r, csw, anti_periodic
        )
    end

    function WilsonEOPreDiracOperator(
        D::WilsonDiracOperator{B,T,C,TF}, U::Gaugefield{B,T}
    ) where {B,T,C,TF}
        check_dims(U, D.temp)
        mass = D.mass
        csw = D.csw
        temp = even_odd(D.temp)
        D_diag = WilsonEODiagonal(temp, mass, csw)
        D_oo_inv = WilsonEODiagonal(temp, D.mass, csw; inverse=true)

        Xμν = C ? Tensorfield(U) : nothing
        calc_diag!(D_diag, D_oo_inv, Xμν, U, mass)

        TF_new = typeof(temp)
        TX = typeof(Xμν)
        TG = typeof(U)
        TO = typeof(D_diag)
        return new{B,T,C,TF_new,TG,TX,TO}(
            U, Xμν, temp, D_diag, D_oo_inv, mass, D.κ, D.r, csw, D.anti_periodic
        )
    end

    function WilsonEOPreDiracOperator(
        D::WilsonEOPreDiracOperator{B,T,C,TF,TG,TX,TO}, U::Gaugefield{B,T}
    ) where {B,T,C,TF,TG,TX,TO}
        check_dims(U, D.temp.parent)
        mass = D.mass
        csw = D.csw
        Xμν = D.Xμν
        temp = D.temp
        D_diag = D.D_diag
        D_oo_inv = D.D_oo_inv

        calc_diag!(D_diag, D_oo_inv, Xμν, U, mass)
        return new{B,T,C,TF,typeof(U),TX,TO}(
            U, Xμν, temp, D_diag, D_oo_inv, mass, D.κ, D.r, csw, D.anti_periodic
        )
    end
end

function (D::WilsonEOPreDiracOperator{B,T})(U::Gaugefield{B,T}) where {B,T}
    return WilsonEOPreDiracOperator(D, U)
end

struct WilsonEODiagonal{B,T,C,A} <: Abstractfield{B,T,A} # So we can overload LinearAlgebra.det on the even-odd diagonal
    U::A
    mass::Float64
    csw::Float64
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors
    ND::Int64 # Number of dirac indices
    function WilsonEODiagonal(
        f::TF, mass, csw; inverse=false
    ) where {B,T,TF<:Union{Fermionfield{B,T},EvenOdd{B,T}}}
        NX, NY, NZ, NT = dims(f)
        _NT = inverse ? NT ÷ 2 : NT
        NC = num_colors(f)
        ND = num_dirac(f)
        NV = NX * NY * NZ * NT
        C = csw == 0 ? false : true
        U = KA.zeros(B(), SMatrix{6,6,Complex{T},36}, 2, NX, NY, NZ, _NT)

        A = typeof(U)
        return new{B,T,C,A}(U, mass, csw, NX, NY, NZ, NT, NV, NC, ND)
    end
end

dims(D_diag::WilsonEODiagonal) = D_diag.NX, D_diag.NY, D_diag.NZ, D_diag.NT
num_colors(D_diag::WilsonEODiagonal) = D_diag.NC

const WilsonEOPreFermionfield{B,T,A} = EvenOdd{B,T,A,4}

struct WilsonEOPreFermionAction{Nf,C,TD,CT,RI1,RI2,RT} <: AbstractFermionAction
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
    function WilsonEOPreFermionAction(
        f::Abstractfield{B,T},
        mass;
        anti_periodic=true,
        r=1,
        csw=0,
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
            power = Nf//4
            rhmc_info_action = RHMCParams(
                power; n=rhmc_order_for_action, precision=rhmc_prec_for_action
            )
            power = Nf//2
            rhmc_info_md = RHMCParams(
                power; n=rhmc_order_for_md, precision=rhmc_prec_for_md
            )
            n_temps = max(rhmc_order_for_md, rhmc_order_for_action)
            rhmc_temps1 = ntuple(_ -> even_odd(Fermionfield(f)), n_temps + 1)
            rhmc_temps2 = ntuple(_ -> even_odd(Fermionfield(f)), n_temps + 1)
        end

        C = csw != 0 ? true : false
        CT = typeof(cg_temps)
        RI1 = typeof(rhmc_info_action)
        RI2 = typeof(rhmc_info_md)
        RT = typeof(rhmc_temps1)
        return new{Nf,C,TD,CT,RI1,RI2,RT}(
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

function Base.show(io::IO, ::MIME"text/plain", S::WilsonEOPreFermionAction{Nf}) where {Nf}
    print(
        io,
        """
        
        |  WilsonEOPreFermionAction(
        |    Nf: $Nf
        |    MASS: $(S.D.mass)
        |    KAPPA: $(S.D.κ)
        |    CSW: $(S.D.csw)
        |    CG TOLERANCE (ACTION): $(S.cg_tol_action)
        |    CG TOLERANCE (MD): $(S.cg_tol_md)
        |    CG MAX ITERS (ACTION): $(S.cg_maxiters_action)
        |    CG MAX ITERS (ACTION): $(S.cg_maxiters_md)
        |    RHMC INFO (Action): $(S.rhmc_info_action)
        |    RHMC INFO (MD): $(S.rhmc_info_md))
        """
    )
    return nothing
end

function Base.show(io::IO, S::WilsonEOPreFermionAction{Nf}) where {Nf}
    print(
        io,
        """
        
        |  WilsonEOPreFermionAction(
        |    Nf: $Nf
        |    MASS: $(S.D.mass)
        |    KAPPA: $(S.D.κ)
        |    CSW: $(S.D.csw)
        |    CG TOLERANCE (ACTION): $(S.cg_tol_action)
        |    CG TOLERANCE (MD): $(S.cg_tol_md)
        |    CG MAX ITERS (ACTION): $(S.cg_maxiters_action)
        |    CG MAX ITERS (ACTION): $(S.cg_maxiters_md)
        |    RHMC INFO (Action): $(S.rhmc_info_action)
        |    RHMC INFO (MD): $(S.rhmc_info_md))
        """
    )
    return nothing
end

function calc_fermion_action(
    fermion_action::WilsonEOPreFermionAction{2},
    U::Gaugefield,
    ϕ_eo::WilsonEOPreFermionfield,
)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψ_eo, temp1, temp2, temp3 = fermion_action.cg_temps
    cg_tol = fermion_action.cg_tol_action
    cg_maxiters = fermion_action.cg_maxiters_action

    clear!(ψ_eo) # initial guess is zero
    solve_dirac!(ψ_eo, DdagD, ϕ_eo, temp1, temp2, temp3, cg_tol, cg_maxiters) # ψ = (D†D)⁻¹ϕ
    Sf = dot(ϕ_eo, ψ_eo)#=  - 2trlog(D.D_diag) =#
    return real(Sf)
end

function calc_fermion_action(
    fermion_action::WilsonEOPreFermionAction{1},
    U::Gaugefield,
    ϕ_eo::WilsonEOPreFermionfield,
)
    error("Not implemented yet")
    # TODO
    return nothing
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
    anti = D.anti_periodic
    D_oo_inv = D.D_oo_inv
    D_diag = D.D_diag

    mul_oe!(ψ_eo, U, ϕ_eo, anti, true, Val(1)) # ψₒ = Dₒₑϕₑ
    mul_oo_inv!(ψ_eo, D_oo_inv) # ψₒ = Dₒₒ⁻¹Dₒₑϕₑ
    mul_eo!(ψ_eo, U, ψ_eo, anti, false, Val(1)) # ψₑ = DₑₒDₒₒ⁻¹Dₒₑϕₑ
    axmy!(D_diag, ϕ_eo, ψ_eo) # ψₑ = Dₑₑϕₑ - DₑₒDₒₒ⁻¹Dₒₑϕₑ
    return nothing
end

function LinearAlgebra.mul!(
    ψ_eo::TF, D::Daggered{WilsonEOPreDiracOperator{CPU,T,C,TF,TG,TX,TO}}, ϕ_eo::TF
) where {T,C,TF,TG,TX,TO}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    check_dims(ψ_eo, ϕ_eo, U)
    anti = D.parent.anti_periodic
    D_oo_inv = D.parent.D_oo_inv
    D_diag = D.parent.D_diag

    mul_oe!(ψ_eo, U, ϕ_eo, anti, true, Val(-1)) # ψₒ = Dₒₑϕₑ
    mul_oo_inv!(ψ_eo, D_oo_inv) # ψₒ = Dₒₒ⁻¹Dₒₑϕₑ
    mul_eo!(ψ_eo, U, ψ_eo, anti, false, Val(-1)) # ψₑ = DₑₒDₒₒ⁻¹Dₒₑϕₑ
    axmy!(D_diag, ϕ_eo, ψ_eo) # ψₑ = Dₑₑϕₑ - DₑₒDₒₒ⁻¹Dₒₑϕₑ
    return nothing
end

function LinearAlgebra.mul!(
    ψ_eo::TF, D::DdaggerD{WilsonEOPreDiracOperator{CPU,T,C,TF,TG,TX,TO}}, ϕ_eo::TF
) where {T,C,TF,TG,TX,TO}
    temp = D.parent.temp
    mul!(temp, D.parent, ϕ_eo) # temp = Dϕ
    mul!(ψ_eo, adjoint(D.parent), temp) # ψ = D†Dϕ
    return nothing
end

function mul_oe!(
    ψ_eo::TF, U::Gaugefield{CPU,T}, ϕ_eo::TF, anti, into_odd, ::Val{dagg}; fac=1
) where {T,TF<:WilsonEOPreFermionfield{CPU,T},dagg}
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
        ψ[_site] = fac * wilson_eo_kernel(U, ϕ, site, anti, T, Val(dagg))
    end
end

function mul_eo!(
    ψ_eo::TF, U::Gaugefield{CPU,T}, ϕ_eo::TF, anti, into_odd, ::Val{dagg}; fac=1
) where {T,TF<:WilsonEOPreFermionfield{CPU,T},dagg}
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
        ψ[_site] = fac * wilson_eo_kernel(U, ϕ, site, anti, T, Val(dagg))
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
    bc⁺ = boundary_factor(anti, site[4], 1, NT)
    bc⁻ = boundary_factor(anti, site[4], -1, NT)
    ψₙ += cmvmul_spin_proj(U[4, site], bc⁺ * ϕ[_siteμ⁺], Val(-4dagg), Val(false))
    ψₙ += cmvmul_spin_proj(U[4, siteμ⁻], bc⁻ * ϕ[_siteμ⁻], Val(4dagg), Val(true))
    return T(0.5) * ψₙ
end

function calc_diag!(
    D_diag::TW, D_oo_inv::TW, ::Nothing, U::Gaugefield{CPU,T}, mass
) where {T,TW<:WilsonEODiagonal{CPU,T,false}}
    check_dims(D_diag, D_oo_inv, U)
    mass_term = Complex{T}(4 + mass)
    fdims = dims(U)
    NV = U.NV
    sz = 2U.NC
    sz2 = sz^2

    #= @batch  =#for site in eachindex(U)
        _site = eo_site(site, fdims..., NV)
        A = SMatrix{sz,sz,Complex{T},sz2}(mass_term * I)

        D_diag[1, site] = A
        D_diag[2, site] = A 

        if isodd(site)
            o_site = switch_sides(_site, fdims..., NV)
            A_inv = SMatrix{sz,sz,Complex{T},sz2}(1/mass_term * I)
            D_oo_inv[1, o_site] = A_inv
            D_oo_inv[2, o_site] = A_inv
        end
    end
end

function calc_diag!(
    D_diag::TW, D_oo_inv::TW, Xμν, U::Gaugefield{CPU,T}, mass
) where {T,TW<:WilsonEODiagonal{CPU,T,true}}
    check_dims(D_diag, D_oo_inv, U)
    mass_term = Complex{T}(4 + mass)
    fdims = dims(U)
    NV = U.NV
    sz = 2U.NC
    sz2 = sz^2
    fac = T(-D_diag.csw / 2)

    fieldstrength_eachsite!(Clover(), Xμν, U)

    #= @batch  =#for site in eachindex(U)
        _site = eo_site(site, fdims..., NV)
        M = SMatrix{sz,sz,Complex{T},sz2}(mass_term * I)

        F₁₂ = Xμν[1, 2, site]
        A₊ = ckron(SMatrix{2,2,Complex{T},4}(view(σ₁₂(T), 1:2, 1:2)), F₁₂)
        A₋ = ckron(SMatrix{2,2,Complex{T},4}(view(σ₁₂(T), 3:4, 3:4)), F₁₂)

        F₁₃ = Xμν[1, 3, site]
        A₊ += ckron(SMatrix{2,2,Complex{T},4}(view(σ₁₃(T), 1:2, 1:2)), F₁₃)
        A₋ += ckron(SMatrix{2,2,Complex{T},4}(view(σ₁₃(T), 3:4, 3:4)), F₁₃)

        F₁₄ = Xμν[1, 4, site]
        A₊ += ckron(SMatrix{2,2,Complex{T},4}(view(σ₁₄(T), 1:2, 1:2)), F₁₄)
        A₋ += ckron(SMatrix{2,2,Complex{T},4}(view(σ₁₄(T), 3:4, 3:4)), F₁₄)

        F₂₃ = Xμν[2, 3, site]
        A₊ += ckron(SMatrix{2,2,Complex{T},4}(view(σ₂₃(T), 1:2, 1:2)), F₂₃)
        A₋ += ckron(SMatrix{2,2,Complex{T},4}(view(σ₂₃(T), 3:4, 3:4)), F₂₃)

        F₂₄ = Xμν[2, 4, site]
        A₊ += ckron(SMatrix{2,2,Complex{T},4}(view(σ₂₄(T), 1:2, 1:2)), F₂₄)
        A₋ += ckron(SMatrix{2,2,Complex{T},4}(view(σ₂₄(T), 3:4, 3:4)), F₂₄)

        F₃₄ = Xμν[3, 4, site]
        A₊ += ckron(SMatrix{2,2,Complex{T},4}(view(σ₃₄(T), 1:2, 1:2)), F₃₄)
        A₋ += ckron(SMatrix{2,2,Complex{T},4}(view(σ₃₄(T), 3:4, 3:4)), F₃₄)

        A₊ = fac * A₊ + M
        A₋ = fac * A₋ + M
        D_diag[1, _site] = A₊ 
        D_diag[2, _site] = A₋ 

        if isodd(site)
            o_site = switch_sides(_site, fdims..., NV)
            D_oo_inv[1, o_site] = cinv(A₊)
            D_oo_inv[2, o_site] = cinv(A₋)
        end
    end
end

function mul_oo_inv!(
    ϕ_eo::WilsonEOPreFermionfield{CPU,T}, D_diag::WilsonEODiagonal{CPU,T}
) where {T}
    check_dims(ϕ_eo, D_diag)
    ϕ = ϕ_eo.parent
    fdims = dims(ϕ)
    NV = ϕ.NV

    #= @batch  =#for _site in eachindex(true, ϕ)
        o_site = switch_sides(_site, fdims..., NV)
        ϕ[o_site] = cmvmul_block(D_diag[1, _site], D_diag[2, _site], ϕ[o_site])
    end

    return nothing
end

function axmy!(
    D_diag::WilsonEODiagonal{CPU,T}, ψ_eo::TF, ϕ_eo::TF
) where {T,TF<:EvenOdd{CPU,T}} # even on even is the default
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    even = true

    #= @batch  =#for _site in eachindex(even, ϕ)
        ϕ[_site] = cmvmul_block(D_diag[1, _site], D_diag[2, _site], ψ[_site]) - ϕ[_site]
    end

    return nothing
end

function trlog(D_diag::WilsonEODiagonal{CPU,T,true}) where {T}
    d = 0.0

    #= @batch reduction=(*, d)  =#for o_site in eachindex(false, D_diag)
    d += log(real(det(D_diag[1, o_site])))
    d += log(real(det(D_diag[2, o_site])))
    end

    return d
end

function trlog(D_diag::WilsonEODiagonal{CPU,T,false}) where {T}
    NC = num_colors(D_diag)
    mass_term = Float64(4 + D_diag.mass)
    logd = 4NC * log(mass_term)
    return D_diag.NV÷2 * logd
end

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
struct WilsonEOPreDiracOperator{B,T,C,TF,TG,TX,TO,BC} <: AbstractDiracOperator{B,T}
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
        U::TG, Fμν::TX, temp::TF, D_diag::TO, D_oo_inv::TO, mass, κ, r, csw, ::Val{C}, bc::BC
    ) where {B,T,C,TG<:Gaugefield{B,T},TX,TF<:WilsonEOPreSpinorfield{B,T},TO,BC}
        return new{B,T,C,TF,TG,TX,TO,BC}(
            U, Fμν, temp, D_diag, D_oo_inv, mass, κ, r, csw, bc
        )
    end

    function WilsonEOPreDiracOperator(
        f::AbstractField{B,T}, mass; bc_str="antiperiodic", r=1, csw=0, kwargs...
    ) where {B,T}
        @assert r == 1 "Only r=1 in Wilson Dirac supported for now"
        κ = 1 / (2mass + 8)
        U = nothing
        C = csw == 0 ? false : true
        hw = C ? 2 : 1
        Fμν = C ? Tensorfield(f) : nothing
        temp = even_odd(Spinorfield(f; hw=hw)) # INFO: Wilson Dirac Op. is 1-hop, so halo_width=1 is enough
        D_diag = Paulifield(temp, csw, false; no_halo=true)
        D_oo_inv = Paulifield(temp, csw, true; no_halo=true)
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
end

# FIXME:
function add_gauge_background(
    D::WilsonEOPreDiracOperator{B,T,C,TF,TG,TX,TO}, U::Gaugefield{B,T}
) where {B,T,C,TF,TG,TX,TO}
    check_dims(U, D.temp.parent)
    mass = D.mass
    csw = D.csw
    Fμν = D.Fμν
    temp = D.temp
    D_diag = D.D_diag
    D_oo_inv = D.D_oo_inv
    bc = D.boundary_condition
    calc_diag!(D_diag, D_oo_inv, Fμν, U, mass)
    return WilsonEOPreDiracOperator(
        U, Fμν, temp, D_diag, D_oo_inv, mass, D.κ, D.r, csw, Val(C), bc
    )
end

@inline default_Nf(::WilsonEOPreDiracOperator) = 2
@inline is_staggered(::WilsonEOPreDiracOperator) = false
@inline has_clover_term(::WilsonEOPreDiracOperator{B,T,C}) where {B,T,C} = C
@inline has_clover_term(::Daggered{W}) where {B,T,C,W<:WilsonEOPreDiracOperator{B,T,C}} = C
@inline has_clover_term(::DdaggerD{W}) where {B,T,C,W<:WilsonEOPreDiracOperator{B,T,C}} = C

# INFO: Need to explicitly define fermion action here, because of the small determinant
function calc_fermion_action(
    fermion_action::FermionAction{true,2,WilsonEOPreDiracOperator},
    U::Gaugefield,
    ϕ_eo::WilsonEOPreSpinorfield,
)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    ψ_eo, temp1, temp2, temp3 = fermion_action.temps[1:4]
    solver_action = fermion_action.solver_action
    tol, maxiters, datafile = get_info(solver_action)

    clear!(ψ_eo) # initial guess is zero
    solve_dirac!(ψ_eo, DdagD, ϕ_eo, temp1, temp2, temp3; tol, maxiters, datafile) # ψ = (D†D)⁻¹ϕ

    Sf = real(dot(ϕ_eo, ψ_eo)) - 2trlog(D.D_diag, D.mass)
    return Sf
end

function calc_fermion_action(
    fermion_action::FermionAction{true,1,WilsonEOPreDiracOperator},
    U::Gaugefield,
    ϕ_eo::WilsonEOPreSpinorfield,
)
    rhmc = fermion_action.rhmc_info_action
    n = get_n(rhmc)
    D = fermion_action.D(U)
    DdagD = DdaggerD(D)
    temp1, temp2 = fermion_action.temps[1:2]
    ψs = fermion_action.temps[3:n+3]
    ps = fermion_action.temps[n+4:2n+4]
    solver_action = fermion_action.solver_action
    tol, maxiters, datafile = get_info(solver_action)

    for v_eo in ψs
        clear!(v_eo)
    end

    shifts = get_β_inverse(rhmc)
    coeffs = get_α_inverse(rhmc)
    α₀ = get_α0_inverse(rhmc)
    solve_dirac_multishift!(ψs, shifts, DdagD, ϕ_eo, temp1, temp2, ps; tol, maxiters, datafile)

    ψ_eo = ψs[1]
    clear!(ψ_eo) # D⁻¹ϕ doesn't appear in the partial fraction decomp so we can use it to sum

    axpy!(α₀, ϕ_eo, ψ_eo)

    for i in 1:n
        axpy!(coeffs[i], ψs[i+1], ψ_eo)
    end

    Sf = real(dot(ψ_eo, ψ_eo)) - 2trlog(D.D_diag, D.mass)
    return Sf
end

function solve_dirac!(
    ψ_eo, D::T, ϕ_eo, temps...; tol=1e-14, maxiters=1000, datafile=""
) where {T<:WilsonEOPreDiracOperator}
    return bicg_stab!(ψ_eo, D, ϕ_eo, temps...; tol, maxiters, datafile)
end

# We overload LinearAlgebra.mul! instead of Gaugefields.mul! so we dont have to import
# The Gaugefields module into CG.jl, which also allows us to use the solvers for 
# for arbitrary arrays, not just fermion fields and dirac operators (good for testing)
function LinearAlgebra.mul!(
    ψ_eo::TF, D::WilsonEOPreDiracOperator{B,T,C,TF,TG,TX,TO}, ϕ_eo::TF
) where {B,T,C,TF,TG,TX,TO}
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
    ψ_eo::TF, D::Daggered{WilsonEOPreDiracOperator{B,T,C,TF,TG,TX,TO,BC}}, ϕ_eo::TF
) where {B,T,C,TF,TG,TX,TO,BC}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
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
    ψ_eo::TF, D::DdaggerD{WilsonEOPreDiracOperator{B,T,C,TF,TG,TX,TO,BC}}, ϕ_eo::TF
) where {B,T,C,TF,TG,TX,TO,BC}
    temp = D.parent.temp
    mul!(temp, D.parent, ϕ_eo) # temp = Dϕ
    mul!(ψ_eo, adjoint(D.parent), temp) # ψ = D†Dϕ
    return nothing
end

function mul_oe!(
    ψ_eo::TF, U::Gaugefield{B,T,M}, ϕ_eo::TF, bc, into_odd, ::Val{dagg}; fac=1
) where {B,T,M,TF<:WilsonEOPreSpinorfield{B,T,M},dagg}
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    bulk = eachindex(ψ)
    odd_half = false
    itr = eachindex(odd_half, ψ, ϕ, U)

    parallelfor(itr, B, Val(M), (U, ϕ_eo), (ψ,), (U, ϕ, ψ)) do o_site, (U, ϕ, ψ)
        site = map_from_half(o_site, bulk)
        _site = into_odd ? o_site : switch_sides(o_site, bulk)
        @inbounds ψ[_site] = fac * wilson_eo_kernel(U, ϕ, site, bc, T, Val(dagg), bulk)
    end

    return nothing
end

function mul_eo!(
    ψ_eo::TF, U::Gaugefield{B,T,M}, ϕ_eo::TF, bc, into_odd, ::Val{dagg}; fac=1
) where {B,T,M,TF<:WilsonEOPreSpinorfield{B,T,M},dagg}
    check_dims(ψ_eo, ϕ_eo, U)
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    bulk = eachindex(ψ)
    even_half = true
    itr = eachindex(even_half, ψ, ϕ, U)

    parallelfor(itr, B, Val(M), (U, ϕ_eo), (ψ,), (U, ϕ, ψ)) do e_site, (U, ϕ, ψ)
        site = map_from_half(e_site, bulk)
        _site = into_odd ? switch_sides(e_site, bulk) : e_site
        @inbounds ψ[_site] = fac * wilson_eo_kernel(U, ϕ, site, bc, T, Val(dagg), bulk)
    end

    return nothing
end

function wilson_eo_kernel(U, ϕ, site, bc, ::Type{T}, ::Val{dagg}, bulk) where {T,dagg}
    # sites that begin with a "_" are meant for indexing into the even-odd preconn'ed
    # fermion field 
    @inbounds begin
        ψₙ = zero(ϕ[site])
        NT = size(U, 4)

        # use @nexprs here to statically generate the loop
        # this makes it so Val(i) is well defined at each iteration and no type-instabilities arise
        @nexprs 4 μ -> (
            Nμ = axes(U, μ);
            _siteμ⁺ = map_to_half(move(site, μ, 1, Nμ), bulk);
            siteμ⁻ = move(site, μ, -1, Nμ);
            _siteμ⁻ = map_to_half(siteμ⁻, bulk);
            ϕ⁺ = apply_bc(ϕ[_siteμ⁺], bc, site, Val(1), NT, Val(μ));
            ϕ⁻ = apply_bc(ϕ[_siteμ⁻], bc, site, Val(-1), NT, Val(μ));
            ψₙ += cmvmul_spin_proj(U[μ, site], ϕ⁺, Val(-μ*dagg), Val(false));
            ψₙ += cmvmul_spin_proj(U[μ, siteμ⁻], ϕ⁻, Val(μ*dagg), Val(true))
        )
    end

    return T(0.5) * ψₙ
end

function calc_diag!(
    D_diag::TW, D_oo_inv::TW, ::Nothing, U::Gaugefield{B,T}, mass
) where {B,T,M,TW<:Paulifield{B,T,M,false}}
    check_dims(D_diag, D_oo_inv, U)
    mass_term = Complex{T}(4 + mass)
    bulk = eachindex(U)
    itr = eachindex(D_diag, D_oo_inv, U)

    parallelfor(itr, B, Val(M), (), (D_diag, D_oo_inv), (D_diag, D_oo_inv)) do site, (D_diag, D_oo_inv)
        _site = map_to_half(site, bulk)
        A = SMatrix{6,6,Complex{T},36}(mass_term * I)
        @inbounds D_diag[site] = PauliMatrix(A, A)

        if isodd(site)
            A_inv = SMatrix{6,6,Complex{T},36}(1/mass_term * I)
            @inbounds D_oo_inv[_site] = PauliMatrix(A_inv, A_inv)
        end
    end
end

function calc_diag!(
    D_diag::TW, D_oo_inv::TW, Fμν::Tensorfield{B,T}, U::Gaugefield{B,T}, mass
) where {B,T,M,TW<:Paulifield{B,T,M,true}} # With clover term
    mass_term = Complex{T}(4 + mass)
    fac = Complex{T}(D_diag.csw / 2)
    bulk = eachindex(U)
    itr = eachindex(D_diag, D_oo_inv, Fμν, U)

    fieldstrength_eachsite!(Clover(), Fμν, U)

    parallelfor(itr, B, Val(M), (), (D_diag, D_oo_inv), (D_diag, D_oo_inv, Fμν)) do site, (D_diag, D_oo_inv, Fμν)
        calc_diag_csw_kernel!(D_diag, D_oo_inv, Fμν, mass_term, site, fac, T, bulk)
    end
end

function calc_diag_csw_kernel!(
    D_diag, D_oo_inv, Fμν, mass_term, site, fac, ::Type{T}, bulk
) where {T}
    _site = map_to_half(site, bulk)
    M = SMatrix{6,6,Complex{T},36}(mass_term * I)
    i = SVector((1, 2))
    j = SVector((3, 4))

    @inbounds begin
        F₁₂ = Fμν[1, site]
        σ = σ12(T)
        A₊ = ckron(σ[i, i], F₁₂)
        A₋ = ckron(σ[j, j], F₁₂)

        F₁₃ = Fμν[2, site]
        σ = σ13(T)
        A₊ += ckron(σ[i, i], F₁₃)
        A₋ += ckron(σ[j, j], F₁₃)

        F₁₄ = Fμν[3, site]
        σ = σ14(T)
        A₊ += ckron(σ[i, i], F₁₄)
        A₋ += ckron(σ[j, j], F₁₄)

        F₂₃ = Fμν[4, site]
        σ = σ23(T)
        A₊ += ckron(σ[i, i], F₂₃)
        A₋ += ckron(σ[j, j], F₂₃)

        F₂₄ = Fμν[5, site]
        σ = σ24(T)
        A₊ += ckron(σ[i, i], F₂₄)
        A₋ += ckron(σ[j, j], F₂₄)

        F₃₄ = Fμν[6, site]
        σ = σ34(T)
        A₊ += ckron(σ[i, i], F₃₄)
        A₋ += ckron(σ[j, j], F₃₄)

        A₊ = fac * A₊ + M
        A₋ = fac * A₋ + M
        D_diag[_site] = PauliMatrix(A₊, A₋)

        if isodd(site)
            D_oo_inv[_site] = PauliMatrix(cinv(A₊), cinv(A₋))
        end
    end

    return nothing
end

function mul_oo_inv!(
    ϕ_eo::WilsonEOPreSpinorfield{B,T,M}, D_oo_inv::Paulifield{B,T}
) where {B,T,M}
    ϕ = ϕ_eo.parent
    odd_half = false
    itr = eachindex(odd_half, ϕ, D_oo_inv)

    parallelfor(itr, B, Val(M), (), (ϕ,), (ϕ, D_oo_inv)) do o_site, (ϕ, D_oo_inv)
        @inbounds ϕ[o_site] = cmvmul_block(D_oo_inv[o_site], ϕ[o_site])
    end

    return nothing
end

function axmy!(
    D_diag::Paulifield{B,T,M}, ψ_eo::TF, ϕ_eo::TF
) where {B,T,M,TF<:WilsonEOPreSpinorfield{B,T}} # even on even is the default
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    even_half = true
    itr = eachindex(even_half, ϕ, ψ, D_diag)

    parallelfor(itr, B, Val(M), (), (ϕ,), (ϕ, ψ, D_diag)) do e_site, (ϕ, ψ, D_diag)
        @inbounds ϕ[e_site] = cmvmul_block(D_diag[e_site], ψ[e_site]) - ϕ[e_site]
    end

    return nothing
end

function trlog(D_diag::Paulifield{B,T,M,false}, mass) where {B,T,M} # Without clover term
    NC = num_colors(D_diag)
    mass_term = Float64(4 + mass)
    logd = 4NC * log(mass_term)
    return length(D_diag)÷2 * logd
end

function trlog(D_diag::Paulifield{B,T,M,true}, ::Any) where {B,T,M} # With clover term
    odd_half = false
    itr = eachindex(odd_half, D_diag)

    d = parallelfor_sum(itr, 0.0, B, Val(M), (), (), (D_diag,)) do dₙ, o_site, (D_diag,)
        p = D_diag[o_site]
        dₙ += log(real(det(p.upper)) * real(det(p.lower)))
    end

    return distributed_reduce(d, +, D_diag)
end

function mul_oe!(
    ψ_eo::TF, U::Gaugefield{B,T}, ϕ_eo::TF, bc, into_odd, ::Val{dagg}; fac=1
) where {B,T,TF<:WilsonEOPreSpinorfield{B,T},dagg}
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    odd_half = false
    bulk = eachindex(ψ)
    itr = eachindex(odd_half, ψ, ϕ, U)
    # TODO: can hide
    update_halo!(U, ϕ)
    @latmap(itr, wilson_oe_gpu!, ψ, U, ϕ, bc, into_odd, fac, T, Val(dagg), bulk)
    return nothing
end

@kernel function wilson_oe_gpu!(
    ψ, @Const(U), @Const(ϕ), bc, into_odd, fac, ::Type{T}, ::Val{dagg}, bulk, itr
) where {T,dagg}
    iglobal = @index(Global, Cartesian)
    o_site = bulk[iglobal]
    site = map_from_half(o_site, bulk)
    _site = into_odd ? o_site : switch_sides(o_site, bulk)
    @inbounds ψ[_site] = fac * wilson_eo_kernel(U, ϕ, site, bc, T, Val(dagg), bulk)
end

function mul_eo!(
    ψ_eo::TF, U::Gaugefield{B,T}, ϕ_eo::TF, bc, into_odd, ::Val{dagg}; fac=1
) where {B,T,TF<:WilsonEOPreSpinorfield{B,T},dagg}
    check_dims(ψ_eo, ϕ_eo, U)
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    even_half = true
    bulk = eachindex(ψ)
    itr = eachindex(even_half, ψ, ϕ, U)
    # TODO: can hide
    update_halo!(U, ϕ)
    @latmap(itr, wilson_eo_gpu!, ψ, U, ϕ, bc, into_odd, fac, T, Val(dagg), bulk)
    return nothing
end

@kernel function wilson_eo_gpu!(
    ψ, @Const(U), @Const(ϕ), bc, into_odd, fac, ::Type{T}, ::Val{dagg}, bulk, itr
) where {T,dagg}
    iglobal = @index(Global, Cartesian)
    e_site = bulk[iglobal]
    site = map_from_half(e_site, bulk)
    _site = into_odd ? switch_sides(e_site, bulk) : e_site
    @inbounds ψ[_site] = fac * wilson_eo_kernel(U, ϕ, site, bc, T, Val(dagg), bulk)
end

function calc_diag!(
    D_diag::TW, D_oo_inv::TW, ::Nothing, U::Gaugefield{B,T}, mass
) where {B<:GPU,T,M,TW<:Paulifield{B,T,M,false}}
    mass_term = Complex{T}(4 + mass)
    bulk = eachindex(D_diag, D_oo_inv, U)
    @latmap(bulk, calc_diag_gpu!, D_diag, D_oo_inv, mass_term)
    return nothing
end

@kernel function calc_diag_gpu!(D_diag, D_oo_inv, mass_term, ::Type{T}, bulk) where {T}
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    calc_diag_kernel!(D_diag, D_oo_inv, mass_term, site, T, bulk)
end

function calc_diag!(
    D_diag::TW, D_oo_inv::TW, Fμν::Tensorfield{B,T}, U::Gaugefield{B,T}, mass
) where {B,T,M,TW<:Paulifield{B,T,M,true}}
    mass_term = Complex{T}(4 + mass)
    fac = Complex{T}(D_diag.csw / 2)
    bulk = eachindex(D_diag, D_oo_inv, Fμν, U)
    # TODO: can hide
    update_halo!(U)

    fieldstrength_eachsite!(Clover(), Fμν, U)

    @latmap(bulk, calc_diag_csw_gpu!, D_diag, D_oo_inv, Fμν, mass_term, fac, T)
    return nothing
end

@kernel function calc_diag_csw_gpu!(
    D_diag, D_oo_inv, @Const(Fμν), mass_term, fac, ::Type{T}, bulk
) where {T}
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    calc_diag_csw_kernel!(D_diag, D_oo_inv, Fμν, mass_term, site, fac, T, bulk)
end

function mul_oo_inv!(
    ϕ_eo::WilsonEOPreSpinorfield{B,T}, D_oo_inv::Paulifield{B,T}
) where {B,T}
    ϕ = ϕ_eo.parent
    even_half = true
    bulk = eachindex(ϕ)
    itr = eachindex(even_half, ϕ, D_oo_inv)
    @latmap(itr, mul_oo_inv_gpu!, ϕ, D_oo_inv, bulk)
    return nothing
end

@kernel function mul_oo_inv_gpu!(ϕ, D_oo_inv, bulk, itr)
    iglobal = @index(Global, Cartesian)
    e_site = itr[iglobal]
    o_site = switch_sides(e_site, bulk)
    ϕ[o_site] = cmvmul_block(D_oo_inv[e_site], ϕ[o_site])
end

function axmy!(D_diag::Paulifield{B,T}, ψ_eo::TF, ϕ_eo::TF) where {B,T,TF}
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent
    even_sites = true
    itr = eachindex(even_sites, ϕ, ψ)
    @latmap(itr, axmy_gpu!, D_diag, ψ, ϕ)
    return nothing
end

@kernel function axmy_gpu!(@Const(D_diag), @Const(ψ), ϕ, itr)
    iglobal = @index(Global, Cartesian)
    e_site = itr[iglobal]
    ϕ[e_site] = cmvmul_block(D_diag[e_site], ψ[e_site]) - ϕ[e_site]
end

function trlog(D_diag::Paulifield{B,T,M,true}, ::Any) where {B,T,M} # With clover term
    odd_half = false
    itr = eachindex(odd_half, D_diag)
    return @latsum(itr, Float64, trlog_kernel, D_diag)
end

@kernel function trlog_kernel(out, @Const(D_diag), itr)
    iblock = @index(Group, Linear)
    iglobal = @index(Global, Cartesian)
    o_site = itr[iglobal]

    d = 0.0
    p = D_diag[o_site]
    d += log(real(det(p.upper)) * real(det(p.lower)))

    out_group = @groupreduce(+, d, 0.0)

    ithread = @index(Local)
    if ithread == 1
        @inbounds out[iblock] = out_group
    end
end

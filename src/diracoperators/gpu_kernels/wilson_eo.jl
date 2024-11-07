function mul_oe!(
    ψ_eo::TF, U::Gaugefield{B,T}, ϕ_eo::TF, bc, into_odd, ::Val{dagg}; fac=1
) where {B,T,TF<:WilsonEOPreSpinorfield{B,T},dagg}
    check_dims(ψ_eo, ϕ_eo, U)
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    fdims = dims(ψ)
    NV = ψ.NV

    @latmap(
        Sequential(), Val(1), wilson_eo_kernel!, ψ, U, ϕ, bc, fdims, NV, fac, T, Val(dagg)
    )
    update_halo!(ψ_eo)
    return nothing
end

@kernel function wilson_eo_kernel!(
    ψ, @Const(U), @Const(ϕ), bc, fdims, NV, fac, ::Type{T}, ::Val{dagg}
) where {T,dagg}
    site = @index(Global, Cartesian)
    _site = if into_odd
        eo_site(site, fdims..., NV)
    else
        eo_site_switch(site, fdims..., NV)
    end

    if iseven(site)
        @inbounds ψ[_site] = fac * wilson_eo_kernel(U, ϕ, site, bc, T, Val(dagg))
    end
end

function mul_eo!(
    ψ_eo::TF, U::Gaugefield{B,T}, ϕ_eo::TF, bc, into_odd, ::Val{dagg}; fac=1
) where {B,T,TF<:WilsonEOPreSpinorfield{B,T},dagg}
    check_dims(ψ_eo, ϕ_eo, U)
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    fdims = dims(ψ)
    NV = ψ.NV

    @latmap(
        Sequential(), Val(1), wilson_oe_kernel!, ψ, U, ϕ, bc, fdims, NV, fac, T, Val(dagg)
    )
    update_halo!(ψ_eo)
    return nothing
end

@kernel function wilson_oe_kernel!(
    ψ, @Const(U), @Const(ϕ), bc, fdims, NV, fac, ::Type{T}, ::Val{dagg}
) where {T,dagg}
    site = @index(Global, Cartesian)
    _site = if into_odd
        eo_site_switch(site, fdims..., NV)
    else
        eo_site(site, fdims..., NV)
    end

    if iseven(site)
        @inbounds ψ[_site] = fac * wilson_eo_kernel(U, ϕ, site, bc, T, Val(dagg))
    end
end

function calc_diag!(
    D_diag::TW, D_oo_inv::TW, ::Nothing, U::Gaugefield{B,T}, mass
) where {B,T,M,TW<:Paulifield{B,T,M,false}}
    check_dims(D_diag, D_oo_inv, U)
    mass_term = Complex{T}(4 + mass)
    fdims = dims(U)
    NV = U.NV

    @latmap(Sequential(), Val(1), calc_diag_kernel!, D_diag, D_oo_inv, mass_term, fdims, NV)
    return nothing
end

@kernel function calc_diag_kernel!(
    D_diag, D_oo_inv, mass_term, fdims, NV, ::Type{T}
) where {T}
    site = @index(Global, Cartesian)
    calc_diag_kernel!(D_diag, D_oo_inv, mass_term, site, fdims, NV, T)
end

function calc_diag!(
    D_diag::TW, D_oo_inv::TW, Fμν::Tensorfield{B,T,M}, U::Gaugefield{B,T,M}, mass
) where {B,T,M,TW<:Paulifield{B,T,M,true}}
    check_dims(D_diag, D_oo_inv, U)
    mass_term = Complex{T}(4 + mass)
    fdims = dims(U)
    NV = U.NV
    fac = Complex{T}(D_diag.csw / 2)

    @latmap(
        Sequential(), Val(1), calc_diag_kernel!, D_diag, D_oo_inv, mass_term, fdims, NV, fac, T
    )
    return nothing
end

@kernel function calc_diag_kernel!(
    D_diag, D_oo_inv, @Const(Fμν), mass_term, fdims, NV, fac, ::Type{T}
) where {T}
    site = @index(Global, Cartesian)
    calc_diag_kernel!(D_diag, D_oo_inv, Fμν, mass_term, site, fdims, NV, fac, T)
end

function mul_oo_inv!(
    ϕ_eo::WilsonEOPreSpinorfield{B,T,M}, D_oo_inv::Paulifield{B,T,M}
) where {B,T,M}
    check_dims(ϕ_eo, D_oo_inv)
    ϕ = ϕ_eo.parent
    fdims = dims(U)
    NV = U.NV

    @latmap(EvenSites(), Val(1), mul_oo_inv_kernel!, ϕ, D_oo_inv, fdims, NV)
    return nothing
end

@kernel function mul_oo_inv_kernel!(ϕ, D_oo_inv, fdims, NV)
    _site = @index(Global, Cartesian)
    o_site = switch_sides(_site, fdims..., NV)
    ϕ[o_site] = cmvmul_block(D_oo_inv[_site], ϕ[o_site])
end

function axmy!(
    D_diag::Paulifield{B,T,M}, ψ_eo::TF, ϕ_eo::TF
) where {B,T,M,TF<:WilsonEOPreSpinorfield{B,T,M}} # even on even is the default
    check_dims(ϕ_eo, ψ_eo)
    ϕ = ϕ_eo.parent
    ψ = ψ_eo.parent

    @latmap(EvenSites(), Val(1), axmy_kernel!, D_diag, ψ, ϕ)
    return nothing
end

@kernel function axmy_kernel!(@Const(D_diag), ψ, ϕ)
    _site = @index(Global, Cartesian)
    ϕ[_site] = cmvmul_block(D_diag[_site], ψ[_site]) - ϕ[_site]
end

function trlog(D_diag::Paulifield{B,T,M,true}, ::Any) where {B,T,M} # With clover term
    fdims = dims(D_diag)
    NV = D_diag.NV
    return @latsum(EvenSites(), Val(1), Float64, trlog_kernel, D_diag, fdims, NV)
end

@kernel function trlog_kernel(out, @Const(D_diag), fdims, NV)
    bi = @index(Group, Linear)
    _site = @index(Global, Cartesian)
    o_site = switch_sides(_site, fdims..., NV)

    resₙ = 0.0
    p = D_diag[o_site]
    resₙ += log(real(det(p.upper)) * real(det(p.lower)))

    out_group = @groupreduce(+, resₙ, 0.0)

    ti = @index(Local)
    if ti == 1
        @inbounds out[bi] = out_group
    end
end

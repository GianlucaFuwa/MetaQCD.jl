function mul_oe!(
    ψ_eo::TF, U::Gaugefield{B,T,M}, ϕ_eo::TF, anti, into_odd, dagg
) where {B<:GPU,T,M,TF<:SpinorfieldEO{B,T,M}}
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    odd_half = false
    bulk = eachindex(ψ)
    halo = M ? ψ.topology.halo_sites : nothing
    itr = eachindex(odd_half, ψ, ϕ, U)
    # TODO: can hide
    update_halo!(U, ϕ)
    @latmap(itr, staggered_oe_gpu!, ψ, U, ϕ, anti, into_odd, dagg, T, halo, bulk)
end

@kernel cpu=false function staggered_oe_gpu!(
    ψ, @Const(U), @Const(ϕ), anti, into_odd, dagg, ::Type{T}, halo, bulk, itr
) where {T}
    iglobal = @index(Global, Cartesian)
    o_site = bulk[iglobal]
    site = map_from_half(o_site, bulk)
    _site = into_odd ? o_site : switch_sides(o_site, bulk)
    @inbounds ψ[_site] = staggered_eo_kernel(U, ϕ, site, anti, T, dagg, halo, bulk)
end

function mul_eo!(
    ψ_eo::TF, U::Gaugefield{B,T,M}, ϕ_eo::TF, anti, into_odd, dagg
) where {B<:GPU,T,M,TF<:SpinorfieldEO{B,T,M}}
    ψ = ψ_eo.parent
    ϕ = ϕ_eo.parent
    even_half = true
    bulk = eachindex(ψ)
    halo = M ? ψ.topology.halo_sites : nothing
    itr = eachindex(even_half, ψ, ϕ, U)
    # TODO: can hide
    update_halo!(U, ϕ)
    @latmap(itr, staggered_eo_gpu!, ψ, U, ϕ, anti, into_odd, dagg, T, halo, bulk)
end

@kernel cpu=false function staggered_eo_gpu!(
    ψ, @Const(U), @Const(ϕ), anti, into_odd, dagg, ::Type{T}, halo, bulk, itr
) where {T}
    iglobal = @index(Global, Cartesian)
    e_site = bulk[iglobal]
    site = map_from_half(e_site, bulk)
    _site = into_odd ? switch_sides(e_site, bulk) : e_site
    @inbounds ψ[_site] = staggered_eo_kernel(U, ϕ, site, anti, T, dagg, halo, bulk)
end

function update!(
    hb::Heatbath{MAXIT,ITR,TOR,NHB,NOR}, U::Gaugefield{B,T,A,GA}; kwargs...
) where {MAXIT,ITR,NHB,TOR,NOR,B<:GPU,T,A,GA}
    @assert ITR!=Sequential
    ALG = eltype(TOR())
    fac_hb = T(U.NC/U.β)
    fac_or = T(-U.β/U.NC)

    hb_kernel! = ITR ≡ Checkerboard2 ? heatbath_C2_kernel! : heatbath_C4_kernel!
    or_kernel! = ITR ≡ Checkerboard2 ? overrelaxation_C2_kernel! : overrelaxation_C4_kernel!

    @latmap(ITR(), NHB(), hb_kernel!, U, GA(), fac_hb, _unwrap_val(MAXIT()))
    numaccepts_or = @latsum(ITR(), NOR(), or_kernel!, U, ALG(), GA(), fac_or)

    U.Sg = calc_gauge_action(U)
    numaccepts = (NOR≡Val{0}) ? 1.0 : numaccepts_or / (4U.NV*_unwrap_val(NOR()))
    return numaccepts
end

@kernel function heatbath_C2_kernel!(U, μ, pass, GA, action_factor, MAXIT)
	iy, iz, it = @index(Global, NTuple)

    for ix in 1+iseven(iy+iz+it+pass):2:size(U, 2)
        site = SiteCoords(ix, iy, iz, it)
        @inbounds old_link = U[μ,site]
        A = staple(GA, U, μ, site)
        @inbounds U[μ,site] = heatbath_SU3(old_link, A, MAXIT, action_factor)
    end
end

@kernel function heatbath_C4_kernel!(U, μ, pass, GA, action_factor, MAXIT)
	iy, iz, it = @index(Global, NTuple)

    for ix in axes(U, 2)
        site = SiteCoords(ix, iy, iz, it)
        if mod1(sum(site.I) + site[μ], 4)==pass
            @inbounds old_link = U[μ,site]
            A = staple(GA, U, μ, site)
            @inbounds U[μ,site] = heatbath_SU3(old_link, A, MAXIT, action_factor)
        end
    end
end

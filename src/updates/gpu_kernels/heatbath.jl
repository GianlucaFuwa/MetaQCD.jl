function update!(hb::Heatbath{Checkerboard2MT,TOR,NHB,NOR},
    U::Gaugefield{GPUD,T,A,GA}; kwargs...) where {NHB,TOR,NOR,T,A,GA}
    @checker2map(heatbath_C2_kernel!, U, GA(), T(U.NC/U.β), hb.MAXIT)

    if NOR !== Val{0}
        fac = T(-U.β/U.NC)
        ALG = eltype(TOR())
        numaccepts = zero(T)
        for _ in 1:_unwrap_val(NOR())
            numaccepts += @checker2reduce(+, overrelaxation_C2_kernel!, U, ALG(), GA(), fac)
        end
    end

    U.Sg = calc_gauge_action(U)
    numaccepts = (NOR≡Val{0}) ? 1.0 : numaccepts / (4U.NV*_unwrap_val(NOR()))
    return numaccepts
end

function update!(hb::Heatbath{Checkerboard4MT,TOR,NHB,NOR},
    U::Gaugefield{GPUD,T,A,GA}; kwargs...) where {NHB,TOR,NOR,T,A,GA}
    @checker4map(heatbath_C4_kernel!, U, GA(), T(U.NC/U.β), hb.MAXIT)

    if NOR !== Val{0}
        fac = T(-U.β/U.NC)
        ALG = eltype(TOR())
        numaccepts = zero(T)
        for _ in 1:_unwrap_val(NOR())
            numaccepts += @checker4reduce(+, overrelaxation_C4_kernel!, U, ALG(), GA(), fac)
        end
    end

    U.Sg = calc_gauge_action(U)
    numaccepts = (NOR≡Val{0}) ? 1.0 : numaccepts / (4*U.NV*_unwrap_val(NOR()))
    return numaccepts
end

@kernel function heatbath_C2_kernel!(U, μ, pass, GA, action_factor, MAXIT)
	iy, iz, it = @index(Global, NTuple)

    @unroll for ix in 1+iseven(iy+iz+it+pass):2:size(U, 2)
        site = SiteCoords(ix, iy, iz, it)
        @inbounds old_link = U[μ,site]
        A = staple(GA, U, μ, site)
        @inbounds U[μ,site] = heatbath_SU3(old_link, A, MAXIT, action_factor)
    end
end

@kernel function heatbath_C4_kernel!(U, μ, pass, GA, action_factor, MAXIT)
	iy, iz, it = @index(Global, NTuple)

    @unroll for ix in 1:size(U, 2)
        site = SiteCoords(ix, iy, iz, it)
        if mod1(sum(site.I) + site[μ], 4)==pass
            @inbounds old_link = U[μ,site]
            A = staple(GA, U, μ, site)
            @inbounds U[μ,site] = heatbath_SU3(old_link, A, MAXIT, action_factor)
        end
    end
end

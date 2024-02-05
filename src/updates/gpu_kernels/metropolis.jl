function update!(metro::Metropolis{Checkerboard2MT,NH,TOR,NOR},
    U::Gaugefield{GPUD,T,A,GA}; kwargs...) where {NH,TOR,NOR,T,A,GA}
    fac = T(-U.β/U.NC)
    ϵ = T(metro.ϵ[])
    hits = _unwrap_val(NH())
    numaccepts_metro = @checker2reduce(+, metropolis_C2_kernel!, U, GA(), ϵ, fac, hits)

    if NOR !== Val{0}
        ALG = eltype(TOR())
        numaccepts = zero(T)
        for _ in 1:_unwrap_val(NOR())
            numaccepts += @checker2reduce(+, overrelaxation_C2_kernel!, U, ALG(), GA(), fac)
        end
    end

    adjust_ϵ!(metro, numaccepts_metro / (4U.NV*hits))
    U.Sg = calc_gauge_action(U)
    numaccepts = (NOR≡Val{0}) ? 1.0 : numaccepts / (4U.NV*_unwrap_val(NOR()))
    return numaccepts
end

function update!(metro::Metropolis{Checkerboard4MT,NH,TOR,NOR},
    U::Gaugefield{GPUD,T,A,GA}; kwargs...) where {NH,TOR,NOR,T,A,GA}
    fac = T(-U.β/U.NC)
    ϵ = T(metro.ϵ[])
    hits = _unwrap_val(NH())
    numaccepts_metro = @checker4reduce(+, metropolis_C4_kernel!, U, GA(), ϵ, fac, hits)

    if NOR !== Val{0}
        ALG = eltype(TOR())
        numaccepts = zero(T)
        for _ in 1:_unwrap_val(NOR())
            numaccepts += @checker4reduce(+, overrelaxation_C4_kernel!, U, ALG(), GA(), fac)
        end
    end

    adjust_ϵ!(metro, numaccepts_metro / (4U.NV*hits))
    U.Sg = calc_gauge_action(U)
    numaccepts = (NOR≡Val{0}) ? 1.0 : numaccepts / (4U.NV*_unwrap_val(NOR()))
    return numaccepts
end

@kernel function metropolis_C2_kernel!(out, U, μ, pass, GA, ϵ, fac, numhits, neutral)
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	iy, iz, it = @index(Global, NTuple)
    numaccepts = neutral

    for ix in 1+iseven(iy+iz+it+pass):2:size(U, 2)
        site = SiteCoords(ix, iy, iz, it)
        A_adj = staple(GA, U, μ, site)'
        @unroll for _ in 1:numhits
            X = gen_SU3_matrix(ϵ, Float32)
            @inbounds old_link = U[μ,site]
            new_link = cmatmul_oo(X, old_link)

            ΔSg = fac * real(multr((new_link - old_link), A_adj))

            accept = rand(Float64) ≤ exp(-ΔSg)

            if accept
                @inbounds U[μ,site] = proj_onto_SU3(new_link)
                numaccepts += 1f0
            end
        end
    end

    out_group = @groupreduce(+, numaccepts, neutral)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] += out_group
	end
end

@kernel function metropolis_C4_kernel!(out, U, μ, pass, GA, ϵ, fac, numhits, neutral)
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
    iy, iz, it = @index(Global, NTuple)
    numaccepts = neutral

    for ix in 1:size(U, 2)
        site = SiteCoords(ix, iy, iz, it)
        if mod1(sum(site.I) + site[μ], 4)==pass
            A_adj = staple(GA, U, μ, site)'
            @unroll for _ in 1:numhits
                X = gen_SU3_matrix(ϵ, Float32)
                @inbounds old_link = U[μ,site]
                new_link = cmatmul_oo(X, old_link)

                ΔSg = fac * real(multr((new_link - old_link), A_adj))

                accept = rand(Float32) ≤ exp(-ΔSg)

                if accept
                    @inbounds U[μ,site] = proj_onto_SU3(new_link)
                    numaccepts += 1f0
                end
            end
        end
    end

    out_group = @groupreduce(+, numaccepts, neutral)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] += out_group
	end
end

@kernel function overrelaxation_C2_kernel!(out, U, μ, pass, ALG, GA, fac, neutral)
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	iy, iz, it = @index(Global, NTuple)
    numaccepts = neutral

    @unroll for ix in 1+iseven(iy+iz+it+pass):2:size(U, 2)
        site = SiteCoords(ix, iy, iz, it)
        A_adj = staple(GA, U, μ, site)'
        @inbounds old_link = U[μ,site]
        new_link = overrelaxation_SU3(ALG, old_link, A_adj)

        ΔSg = fac * real(multr(new_link - old_link, A_adj))
        accept = (rand(Float32) < exp(-ΔSg))

        if accept
            @inbounds U[μ,site] = new_link
            numaccepts += 1f0
        end
    end

    out_group = @groupreduce(+, numaccepts, neutral)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] += out_group
	end
end

@kernel function overrelaxation_C4_kernel!(out, U, μ, pass, ALG, GA, fac, neutral)
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	iy, iz, it = @index(Global, NTuple)
    numaccepts = neutral

    @unroll for ix in 1:size(U, 2)
        site = SiteCoords(ix, iy, iz, it)
        if mod1(sum(site.I) + site[μ], 4)==pass
            A_adj = staple(GA, U, μ, site)'

            @inbounds old_link = U[μ,site]
            new_link = overrelaxation_SU3(ALG, old_link, A_adj)

            ΔSg = fac * real(multr(new_link - old_link, A_adj))
            accept = (rand(Float32) < exp(-ΔSg))

            if accept
                @inbounds U[μ,site] = new_link
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

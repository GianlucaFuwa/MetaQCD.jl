function update!(metro::Metropolis{ITR,NH,TOR,NOR}, U::Gaugefield{B,T,A,GA};
                 kwargs...) where {ITR,NH,TOR,NOR,B<:GPU,T,A,GA}
    fac = T(-U.β/U.NC)
    ϵ = T(metro.ϵ[])
    hits = _unwrap_val(NH())
    ALG = eltype(TOR())

    metro_kernel! = ITR ≡ Checkerboard2 ? metropolis_C2_kernel! : metropolis_C4_kernel!
    or_kernel! = ITR ≡ Checkerboard2 ? overrelaxation_C2_kernel! : overrelaxation_C4_kernel!

    numaccepts_metro = @latsum(ITR(), Val(1), metro_kernel!, U, GA(), ϵ, fac, hits)
    numaccepts_or = @latsum(ITR(), NOR(), or_kernel!, U, ALG(), GA(), fac)

    numaccepts_metro /= 4U.NV*hits
    @level2("|  Metro acceptance: $(numaccepts_metro)")
    adjust_ϵ!(metro, numaccepts_metro)
    U.Sg = calc_gauge_action(U)
    numaccepts = (NOR≡Val{0}) ? 1.0 : numaccepts_or / (4U.NV*_unwrap_val(NOR()))
    return numaccepts
end

@kernel function metropolis_C2_kernel!(out, U, μ, pass, GA, ϵ, fac, numhits)
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	iy, iz, it = @index(Global, NTuple)
    numaccepts = 0i32

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
                numaccepts += 1i32
            end
        end
    end

    out_group = @groupreduce(+, numaccepts, 0i32)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] += out_group
	end
end

@kernel function metropolis_C4_kernel!(out, U, μ, pass, GA, ϵ, fac, numhits)
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
    iy, iz, it = @index(Global, NTuple)
    numaccepts = 0i32

    for ix in axes(U, 2)
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
                    numaccepts += 1i32
                end
            end
        end
    end

    out_group = @groupreduce(+, numaccepts, 0i32)

	ti = @index(Local)
	if ti == 1
		@inbounds out[bi] += out_group
	end
end

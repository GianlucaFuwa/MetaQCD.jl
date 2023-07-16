struct MetroUpdate <: AbstractUpdate
    eo::Bool
	ϵ::Base.RefValue{Float64}
	multi_hit::Int64
	target_acc::Float64
    numOR::Int64

	function MetroUpdate(U, eo, ϵ, multi_hit, target_acc, numOR)
		m_ϵ = Base.RefValue{Float64}(ϵ)
		return new(eo, m_ϵ, multi_hit, target_acc, numOR)
	end
end

function update!(updatemethod, U, verbose::VerboseLevel; Bias = nothing, metro_test = true)
    numOR = updatemethod.numOR

	if updatemethod.eo
		numaccepts_metro = metro_sweep_eo!(U, updatemethod, metro_test = true)
	else
		numaccepts_metro = metro_sweep!(U, updatemethod, metro_test = true)
	end

    numaccepts_metro /= U.NV * 4 * updatemethod.multi_hit
    adjust_ϵ!(updatemethod, numaccepts_metro)

    numaccepts_or = 0.0

    for _ in 1:numOR
        if updatemethod.eo
            numaccepts_or += overrelaxation_sweep_eo!(U, metro_test = true)
        else
            numaccepts_or += overrelaxation_sweep!(U, metro_test = true)
        end
    end

    numaccepts_or /= numOR > 0 ? U.NV * 4 * numOR : 1.0
    normalize!(U)
	return numaccepts_metro + numaccepts_or
end

function metro_sweep!(U::Gaugefield{GA}, metro; metro_test = true) where {GA}
	ϵ = metro.ϵ[]
	multi_hit = metro.multi_hit
	numaccept = 0.0
	staple = GA()
    action_factor = -U.β / 3

    for μ in 1:4
	    for site in eachindex(U)
            A_adj = staple(U, μ, site)'

            for _ in 1:multi_hit
                X = gen_SU3_matrix(ϵ)
                old_link = U[μ][site]
                new_link = cmatmul_oo(X, old_link)

                ΔS = action_factor * real(multr((new_link - old_link), A_adj))

                accept = metro_test ? (rand() ≤ exp(-ΔS)) : true

                if accept
                    U[μ][site] = new_link
                    numaccept += accept
                end
            end

        end
	end

	return numaccept
end

function metro_sweep_eo!(U::Gaugefield{GA}, metro; metro_test = true) where {GA}
    NX, NY, NZ, NT = size(U)
	ϵ = metro.ϵ[]
	multi_hit = metro.multi_hit
	spacing = 8
    numaccepts = zeros(Float64, nthreads() * spacing)
    staple = GA()
    action_factor = -U.β / 3

	for μ in 1:4
        for pass in 1:2
            @threads for it in 1:NT
                for iz in 1:NZ
                    for iy in 1:NY
                        for ix in 1+mod(it + iz + iy, pass):2:NX
                            site = SiteCoords(ix, iy, iz, it)
                            A_adj = staple(U, μ, site)'

                            for _ in 1:multi_hit
                                X = gen_SU3_matrix(ϵ)
                                link = U[μ][site]
                                XU = cmatmul_oo(X, link)

                                ΔSg = action_factor * real(multr((XU - link), A_adj))

                                accept = metro_test ? (rand() ≤ exp(-ΔSg)) : true

                                if accept
                                    U.Sg += ΔSg
                                    U[μ][site] = XU
                                    numaccept[threadid() * spacing] += accept
                                end
                            end

                        end
                    end
                end
            end
        end
	end

	return sum(numaccepts)
end

function adjust_ϵ!(metro, numaccepts)
	metro.ϵ[] += (numaccepts - metro.target_acc) * 0.2
    metro.ϵ[] = min(1.0, metro.ϵ[])
	return nothing
end

struct Metropolis{ITR,TOR} <: AbstractUpdate
	ϵ::Base.RefValue{Float64}
	numhits::Int64
	target_acc::Float64
    OR::TOR
    numOR::Int64

	function Metropolis(
        ::Gaugefield{GA},
        eo, ϵ, numhits, target_acc, or_alg, numOR;
        verbose = nothing,
    ) where {GA}
        println_verbose1(verbose, ">> Setting Metropolis...")
		m_ϵ = Base.RefValue{Float64}(ϵ)

        if eo
            println_verbose1(verbose, "\t>> PARALLELIZATION ENABLED")
            if GA==WilsonGaugeAction
                ITR = Checkerboard2MT
            else
                ITR = Checkerboard4MT
            end
        else
            println_verbose1(verbose, "\t>> PARALLELIZATION DISABLED")
            if GA==WilsonGaugeAction
                ITR = Checkerboard2
            else
                ITR = Checkerboard4
            end
        end
        println_verbose1(verbose, "\t>> ITERATOR = $(ITR)")

        OR = Overrelaxation(or_alg)
        TOR = typeof(OR)
        println_verbose1(verbose, "\t>> NUMBER OF METROPOLIS HITS = $(numhits)")
        println_verbose1(verbose, "\t>> TARGET ACCEPTANCE RATE = $(target_acc)")
        println_verbose1(verbose, "\t>> OVERRELAXATION ALGORITHM = $(TOR)")
        println_verbose1(verbose, "\t>> NUM. OF OVERRELAXATION SWEEPS = $(numOR)\n")

		return new{ITR, typeof(OR)}(m_ϵ, numhits, target_acc, OR, numOR)
	end
end

function update!(metro::Metropolis{ITR}, U, ::VerboseLevel; kwargs...) where {ITR}
    numaccepts = 0

    numaccepts += sweep_reduce!(ITR(), 1, metro, U, -U.β/U.NC)
    normalize!(U)
    numaccepts += sweep_reduce!(ITR(), metro.numOR, metro.OR, U, -U.β/U.NC)
    normalize!(U)

    numaccepts /= (4*U.NV*metro.numhits + 4*U.NV*metro.numOR)
    adjust_ϵ!(metro, numaccepts)
	return numaccepts
end

function (metro::Metropolis)(U, μ, site, action_factor, metro_test=true)
    A_adj = staple(U, μ, site)'
    numaccepts = 0

    for _ in 1:metro.numhits
        X = gen_SU3_matrix(metro.ϵ[])
        old_link = U[μ][site]
        new_link = cmatmul_oo(X, old_link)

        ΔSg = action_factor * real(multr((new_link - old_link), A_adj))

        accept = metro_test ? (rand(Float64)≤exp(-ΔSg)) : true

        if accept
            U[μ][site] = new_link
            U.Sg += ΔSg
            numaccepts += 1
        end
    end
    return numaccepts
end

function adjust_ϵ!(metro, numaccepts)
	metro.ϵ[] += (numaccepts - metro.target_acc) * 0.2 # 0.2 is arbitrarily chosen
	return nothing
end

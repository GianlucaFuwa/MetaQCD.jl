struct Metropolis{ITR,NH,TOR,NOR} <: AbstractUpdate
	ϵ::Base.RefValue{Float64}
	numhits::Int64
	target_acc::Float64
    OR::TOR
    numOR::Int64

	function Metropolis(::Gaugefield{D,T,A,GA}, eo, ϵ, numhits, target_acc, or_alg,
        numOR) where {D,T,A,GA}
        @level1("┌ Setting Metropolis...")
		m_ϵ = Base.RefValue{Float64}(ϵ)

        if eo || D==GPUD
            @level1("|  PARALLELIZATION ENABLED")
            if GA==WilsonGaugeAction
                ITR = Checkerboard2MT
            else
                ITR = Checkerboard4MT
            end
        else
            @level1("|  PARALLELIZATION DISABLED")
            if GA==WilsonGaugeAction
                ITR = Checkerboard2
            else
                ITR = Checkerboard4
            end
        end
        @level1("|  ITERATOR: $(ITR)")

        OR = Overrelaxation(or_alg)
        TOR = typeof(OR)
        @level1("|  NUMBER OF METROPOLIS HITS: $(numhits)")
        @level1("|  TARGET ACCEPTANCE RATE: $(target_acc)")
        @level1("|  OVERRELAXATION ALGORITHM: $(TOR)")
        @level1("|  NUM. OF OVERRELAXATION SWEEPS: $(numOR)")
        @level1("└\n")
		return new{ITR,Val{numhits},TOR,Val{numOR}}(m_ϵ, numhits, target_acc, OR, numOR)
	end
end

function update!(metro::Metropolis{ITR,NH,TOR,NOR}, U; kwargs...) where {ITR,NH,TOR,NOR}
    numaccepts_metro = 0.0
    numaccepts_or = 0.0

    numaccepts_metro += sweep_reduce!(ITR(), Val(1), metro, U, -U.β/U.NC)
    numaccepts_or += sweep_reduce!(ITR(), NOR(), TOR(), U, -U.β/U.NC)

    numaccepts_metro /= 4*U.NV*_unwrap_val(NH())
    @level3("|  Metro acceptance: $(numaccepts_metro)")
    numaccepts_or /= 4*U.NV*_unwrap_val(NOR())
    adjust_ϵ!(metro, numaccepts_metro)
    U.Sg = calc_gauge_action(U)
	return numaccepts_or
end

function (metro::Metropolis{ITR,NH,TOR,NOR})(U::Gaugefield{CPUD,T}, μ, site,
    action_factor) where {ITR,NH,TOR,NOR,T}
    A_adj = staple(U, μ, site)'
    numaccepts = 0

    for _ in 1:_unwrap_val(NH())
        X = gen_SU3_matrix(metro.ϵ[], T)
        old_link = U[μ,site]
        new_link = cmatmul_oo(X, old_link)

        ΔSg = action_factor * real(multr((new_link - old_link), A_adj))

        accept = rand(T) ≤ exp(-ΔSg)

        if accept
            U[μ,site] = proj_onto_SU3(new_link)
            numaccepts += 1
        end
    end
    return numaccepts
end

function adjust_ϵ!(metro, numaccepts)
	metro.ϵ[] += (numaccepts - metro.target_acc) * 0.01 # 0.2 is arbitrarily chosen
	return nothing
end

"""
    Metropolis(U::Gaugefield{B,T,A,GA}, eo, ϵ, numhits, target_acc, or_alg, numorelax) where {B,T,A,GA}

Create a `Metropolis` object.

# Arguments
- `U::Gaugefield{B,T,A,GA}`: Gauge field object.
- `eo`: Even-odd preconditioning.
- `ϵ`: Step size for the update.
- `numhits`: Number of Metropolis hits.
- `target_acc`: Target acceptance rate.
- `or_alg`: Overrelaxation algorithm.
- `numorelax`: Number of overrelaxation sweeps.

# Returns
A Metropolis object with the specified parameters. The gauge action `GA` of the field `U`
determines the iterator used. For the plaquette or Wilson action it uses a
Checkerboard iterator and for rectangular actions it partitions the lattice into four
sublattices.
"""
struct Metropolis{ITR,NH,TOR,NOR} <: AbstractUpdate
    ϵ::Base.RefValue{Float64}
    numhits::Int64
    target_acc::Float64
    overrelaxation::TOR
    numorelax::Int64
    function Metropolis(
        ::Gaugefield{B,T,A,GA}, ϵ, numhits, target_acc, or_alg, numorelax
    ) where {B,T,A,GA}
        @level1("┌ Setting Metropolis...")
        m_ϵ = Base.RefValue{Float64}(ϵ)
        ITR = (GA == WilsonGaugeAction) ? Checkerboard2 : Checkerboard4

        orelax = Overrelaxation(or_alg)
        TOR = typeof(orelax)
        @level1("|  ITERATOR: $(string(ITR))")
        @level1("|  NUMBER OF METROPOLIS HITS: $(numhits)")
        @level1("|  TARGET ACCEPTANCE RATE: $(target_acc)")
        @level1("|  OVERRELAXATION ALGORITHM: $(string(TOR))")
        @level1("|  NUM. OF OVERRELAXATION SWEEPS: $(numorelax)")
        @level1("└\n")
        return new{ITR,Val{numhits},TOR,Val{numorelax}}(m_ϵ, numhits, target_acc, orelax, numorelax)
    end
end

function update!(metro::Metropolis{ITR,NH,TOR,NOR}, U; kwargs...) where {ITR,NH,TOR,NOR}
    fac = -U.β / U.NC
    GA = gauge_action(U)
    numaccepts_metro = @latsum(ITR(), Val(1), metro, U, GA(), fac)
    numaccepts_or = @latsum(ITR(), NOR(), TOR(), U, GA(), fac)

    numaccepts_metro /= 4 * U.NV * _unwrap_val(NH())
    @level3("|  Metro acceptance: $(numaccepts_metro)")
    adjust_ϵ!(metro, numaccepts_metro)
    U.Sg = calc_gauge_action(U)
    numaccepts = (NOR ≡ Val{0}) ? 1.0 : numaccepts_or / (4U.NV * _unwrap_val(NOR()))
    return numaccepts
end

function (metro::Metropolis{ITR,NH})(U, μ, site, GA, action_factor) where {ITR,NH}
    T = float_type(U)
    A_adj = staple(GA, U, μ, site)'
    numaccepts = 0

    for _ in 1:_unwrap_val(NH())
        X = gen_SU3_matrix(metro.ϵ[], T)
        old_link = U[μ, site]
        new_link = cmatmul_oo(X, old_link)

        ΔSg = action_factor * real(multr((new_link - old_link), A_adj))

        accept = rand() ≤ exp(-ΔSg)

        if accept
            U[μ, site] = proj_onto_SU3(new_link)
            numaccepts += 1
        end
    end

    return numaccepts
end

function adjust_ϵ!(metro, numaccepts)
    metro.ϵ[] += (numaccepts - metro.target_acc) * 0.1 # 0.1 is arbitrarily chosen
    return nothing
end

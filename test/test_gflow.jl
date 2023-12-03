function SU3testgradflow()
    Random.seed!(123)
    println("SU3testgradflow")
    NX = 8; NY = 8; NZ = 8; NT = 8;
    U = random_gauges(NX, NY, NZ, NT, 5.7, WilsonGaugeAction)
    mfac = 1 / (6 * U.NV * U.NC)
    plaq = plaquette_trace_sum(U) * mfac
    println("0\tplaq: $plaq")

    g = GradientFlow(U, "euler", 3, 1, 0.12)
    s = StoutSmearing(U, 3, 0.12)

    MetaQCD.Gaugefields.substitute_U!(g.Uflow, U)

    for iflow in 1:g.numflow
        flow!(g)
        plaq = plaquette_trace_sum(g.Uflow) * mfac
        println("$iflow\tplaq (gflow): $plaq")
    end

    calc_smearedU!(s, U)
    plaqs = plaquette_trace_sum(s.Usmeared_multi[end]) * mfac
    println("3\tplaq (stout): $plaqs")

    return isapprox(plaq, plaqs)
end

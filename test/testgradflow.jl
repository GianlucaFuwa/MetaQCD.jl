function SU3testgradflow()
    Random.seed!(123)
    println("SU3testgradflow")
    NX = 8; NY = 8; NZ = 8; NT = 8;
    U = random_gauges(NX, NY, NZ, NT, 5.7, type_of_gaction=WilsonGaugeAction)
    mfac = 1 / (6 * U.NV * U.NC)
    plaq = plaquette_trace_sum(U) * mfac
    println("0\tplaq: $plaq")

    g = GradientFlow(U, integrator="euler", numflow=3, steps=1, tf=0.12, measure_every=1)
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

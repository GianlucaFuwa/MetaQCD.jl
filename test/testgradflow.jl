function testgradflow(NX, NY, NZ, NT)
    Random.seed!(123)

    U = random_gauges(NX, NY, NZ, NT, 5.7, type_of_gaction = WilsonGaugeAction)
    mfac = 1 / (6 * U.NV * U.NC)
    plaq = plaquette_trace_sum(U) * mfac
    println("0\tplaq: $plaq")

    g = GradientFlow(
        U,
        integrator = "euler",
        numflow = 3,
        steps = 1,
        ϵ = 0.12,
        measure_every = 1,
    )

    s = StoutSmearing(U, 3, 0.12)

    MetaQCD.Gaugefields.substitute_U!(g.Uflow, U)

    for iflow in 1:g.numflow
        flow!(g)
        plaq = plaquette_trace_sum(g.Uflow) * mfac
        println("$iflow\tplaq: $plaq")
    end

    calc_smearedU!(s, U)
    plaqs = plaquette_trace_sum(s.Usmeared_multi[end]) * mfac
    println("3\tplaq: $plaqs")

    return plaq, plaqs
end

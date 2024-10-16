function test_gradflow(backend=CPU)
    Random.seed!(123)
    println("SU3testgradflow")
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    U = Gaugefield{CPU,Float64,WilsonGaugeAction}(NX, NY, NZ, NT, 6.0)
    filename = pkgdir(MetaQCD, "test", "testconf.txt")
    load_config!(BridgeFormat(), U, filename)
    if backend !== CPU
        U = MetaQCD.to_backend(backend, U)
    end

    mfac = 1 / (6 * U.NV * U.NC)
    plaq = plaquette_trace_sum(U) * mfac

    g = GradientFlow(U, "euler", 3, 1, 0.12)
    s = StoutSmearing(U, 3, 0.12)

    copy!(g.Uflow, U)

    println("0\tplaq: $plaq")
    for iflow in 1:g.numflow
        flow!(g)
        plaq = plaquette_trace_sum(g.Uflow) * mfac
        println("$iflow\tplaq (gflow): $plaq")
    end

    calc_smearedU!(s, U)
    plaqs = plaquette_trace_sum(s.Usmeared_multi[end]) * mfac
    println("3\tplaq (stout): $plaqs\n")

    return isapprox(plaq, plaqs)
end

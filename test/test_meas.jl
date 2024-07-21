function test_measurements(backend=CPU)
    println("SU3testmeas")
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    U = Gaugefield{backend,Float64,WilsonGaugeAction}(NX, NY, NZ, NT, 6.0)
    filename = "./test/testconf.txt"
    loadU!(BridgeFormat(), U, filename);
    if !isnothing(backend)
        U = MetaQCD.to_backend(backend, U)
    end

    m_plaq = PlaquetteMeasurement(U)
    @time plaq = measure(m_plaq, U)
    println("PLAQUETTE: $plaq")

    println("==========")

    m_poly = PolyakovMeasurement(U)
    @time poly = measure(m_poly, U)
    println("POLYAKOV: $poly")

    println("==========")

    m_wilson = WilsonLoopMeasurement(U)
    @time wilsonloop = measure(m_wilson, U)
    println("WILSONLOOP: ", wilsonloop)

    println("==========")

    TC_methods  = ["plaquette", "clover", "improved"]
    m_topo = TopologicalChargeMeasurement(U, TC_methods=TC_methods)
    @time topo = measure(m_topo, U)
    println("TOPCHARGE: $topo")

    println("==========")

    ED_methods  = ["plaquette", "clover", "improved"]
    m_ed = EnergyDensityMeasurement(U, ED_methods=ED_methods)
    @time ed = measure(m_ed, U)
    println("ENERGYDENSITY: $ed")

    println("==========")

    GA_methods = ["wilson", "symanzik_tree", "symanzik_tad", "iwasaki", "dbw2"]
    m_gaction = GaugeActionMeasurement(U, GA_methods=GA_methods)
    @time gaction = measure(m_gaction, U)
    println("GAUGEACTION: $gaction\n")
    return plaq, poly, topo["plaquette"], topo["clover"], topo["improved"], wilsonloop[1, 1]
end

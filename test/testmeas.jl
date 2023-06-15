using MetaQCD

function SU3testmeas()
    println("SU3testmeas")
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    U = identity_gauges(NX, NY, NZ, NT, 5.7)
    filename = "./test/testconf.txt"
    load_BridgeText!(filename, U)

    m_plaq = PlaquetteMeasurement(U)
    @time plaq = get_value(measure(m_plaq, U))
    println("plaq: $plaq")

    println("==========")

    m_poly = PolyakovMeasurement(U)
    @time poly = get_value(measure(m_poly, U))
    println("poly: $poly")

    println("==========")

    m_wilson = WilsonLoopMeasurement(U)
    @time wilsonloop = get_value(measure(m_wilson, U))
    println("wilson loop: ", wilsonloop)

    println("==========")

    TC_methods  = ["plaquette", "clover", "improved"]
    m_topo = TopologicalChargeMeasurement(U, TC_methods=TC_methods)
    @time topo = get_value(measure(m_topo, U))
    println("topo: $topo")

    println("==========")

    GA_methods = ["wilson", "symanzik_tree", "symanzik_tad", "iwasaki", "dbw2"]
    m_gaction = GaugeActionMeasurement(U, GA_methods=GA_methods)
    @time gaction = get_value(measure(m_gaction, U))
    println("gaction: $gaction")

    return plaq, poly, topo["plaquette"], topo["clover"], topo["improved"], wilsonloop[1]
end
# SU3test()

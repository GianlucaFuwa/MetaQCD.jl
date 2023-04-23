include("../src/system/MetaQCD.jl")
using .MetaQCD
function SU3test()
    println(SU3test)
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    U = IdentityGauges(NX, NY, NZ, NT, 5.7)
    filename = "./test/testconf.txt"
    load_BridgeText!(filename, U)

    m_plaq = Plaquette_measurement(U)
    m_poly = Polyakov_measurement(U)
    plaq = get_value(measure(m_plaq, U))
    poly = get_value(measure(m_poly, U))
    println("plaq: $plaq")
    println("poly: $poly")

    m_wilson = Wilson_loop_measurement(U)
    wilsonloop = get_value(measure(m_wilson, U))
    println("wilson loop: ", wilsonloop)

    TC_methods  = ["Plaquette", "Clover", "Improved"]
    m_topo = Topological_charge_measurement(U, TC_methods=TC_methods)
    topo = get_value(measure(m_topo, U))
    println("topo: $topo")

    GA_methods = ["Wilson", "Symanzik", "Iwasaki", "DBW2"]
    m_gaction = Gauge_action_measurement(U, GA_methods=GA_methods)
    gaction = get_value(measure(m_gaction, U))
    println("gaction: $gaction")
end
SU3test()
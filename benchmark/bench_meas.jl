module BenchMeas

using BenchmarkTools
using MetaQCD

import Random

Random.seed!(1234)

N = 16

suite = BenchmarkGroup()
s = suite["measurements"] = BenchmarkGroup()

for T in (Float32, Float64)
    U = Gaugefield{CPU,T,WilsonGaugeAction}(N, N, N, N, 6.0)
    random_gauges!(U)

    m_plaq = PlaquetteMeasurement(U)
    m_poly = PolyakovMeasurement(U)
    m_wilson = WilsonLoopMeasurement(U)
    TC_methods  = ["plaquette", "clover", "improved"]
    m_topo = TopologicalChargeMeasurement(U, TC_methods=TC_methods)
    ED_methods  = ["plaquette", "clover", "improved"]
    m_ed = EnergyDensityMeasurement(U, ED_methods=ED_methods)
    GA_methods = ["wilson", "symanzik_tree", "iwasaki", "dbw2"]
    m_gaction = GaugeActionMeasurement(U, GA_methods=GA_methods)

    s["$(T), plaq"] = @benchmarkable(measure($m_plaq, $U, 1, 1))
    s["$(T), poly"] = @benchmarkable(measure($m_poly, $U, 1, 1))
    s["$(T), wilson"] = @benchmarkable(measure($m_wilson, $U, 1, 1))
    s["$(T), topo"] = @benchmarkable(measure($m_topo, $U, 1, 1))
    s["$(T), ed"] = @benchmarkable(measure($m_ed, $U, 1, 1))
    s["$(T), gaction"] = @benchmarkable(measure($m_gaction, $U, 1, 1))
end

end

BenchMeas.suite


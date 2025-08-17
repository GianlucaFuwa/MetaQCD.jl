module BenchMeas

using BenchmarkTools
using MetaQCD

import Random

MetaQCD.MetaIO.set_global_logger!(1, nothing; tc=false) # INFO: disable logging during benchmarks
Random.seed!(1234)

N = 16

suite = BenchmarkGroup()
s = suite["measurements"] = BenchmarkGroup()

for T in (Float32, Float64)
    U = Gaugefield{CPU,T,WilsonGaugeAction,12}(N, N, N, N, 6.0)
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

    s["Avg Plaquette, $(T)"] = @benchmarkable(measure($m_plaq, $U))
    s["Polyakov Loop, $(T)"] = @benchmarkable(measure($m_poly, $U))
    s["Wilson Loops (2x2 + 4x4), $(T)"] = @benchmarkable(measure($m_wilson, $U))
    s["Top. Charge (Plaq + Clov + Imp), $(T)"] = @benchmarkable(measure($m_topo, $U))
    s["Energy Density, $(T)"] = @benchmarkable(measure($m_ed, $U))
    s["Gauge Action (W + LW + IW + DBW2), $(T)"] = @benchmarkable(measure($m_gaction, $U))
end

end

BenchMeas.suite


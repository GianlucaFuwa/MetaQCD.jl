module BenchGauge

using BenchmarkTools
using MetaQCD

import Random

MetaQCD.MetaIO.set_global_logger!(1, nothing; tc=false) # INFO: disable logging during benchmarks

actions = (
    WilsonGaugeAction,
    SymanzikTreeGaugeAction,
)

Random.seed!(1234)

N = 16

suite = BenchmarkGroup()

for GA in actions
    s = suite["1HB + 4OR"] = BenchmarkGroup()
    for T in (Float32, Float64)
        U = Gaugefield{CPU,T,GA}(N, N, N, N, 6.0)
        hb = Heatbath(U, 100, 1, Subgroups, 4)
        random_gauges!(U)

        s["$(T), $(GA)"] = @benchmarkable(update!($hb, $U))
    end
end

end

BenchGauge.suite


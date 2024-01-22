using MetaQCD
using JET
using Random

Random.seed!(1206)

L = (16, 16, 16, 16)
β = 6.1142
gaction = WilsonGaugeAction

U = MetaQCD.initial_gauges("hot", L..., β, gaction);
bias = Bias(Clover(), StoutSmearing(U, 5, 0.12), true, Parametric((-5, 5), 10, 0, 100, 1.4),
            "", 0, [""], nothing);

metro = Metropolis(U, true, 0.1, 10, 0.5, "subgroups", 4)
hb = Heatbath(U, true, 10, 1, "subgroups", 4)
hmc = HMC(U, "OMF4", 1, 5);
metahmc = HMC(U, "OMF4", 1, 5; bias_enabled=true);

@test_opt target_modules=(MetaQCD.Updates, MetaQCD.Gaugefields, MetaQCD.Utils,) update!(hb, U)
@test_opt target_modules=(MetaQCD.Updates, MetaQCD.Gaugefields, MetaQCD.Utils,) update!(hmc, U)
@test_opt target_modules=(MetaQCD.Updates, MetaQCD.Gaugefields, MetaQCD.Utils,) update!(metahmc, U, bias=bias)

using MetaQCD
using Random

Random.seed!(1206)

L = (12, 12, 12, 12)
β = 6.0
gaction = WilsonGaugeAction

U = MetaQCD.initial_gauges("hot", L..., β, gaction);
hb = Heatbath(U, true, 10, 1, "subgroups", 4)

function test(hb, U, N)
    for i in 1:N
        update!(hb, U)
        MetaQCD.is_special_unitary(U[1][1], prec=1e-6) || return i
    end
    return N
end

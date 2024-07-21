using MetaQCD
using Random

function test_normality()
    Random.seed!(1206)
    N = 100

    NX = NY = NZ = NT = 4
    β = 6.0
    gaction = WilsonGaugeAction

    U = Gaugefield{CPU,Float64,gaction}(NX, NY, NZ, NT, β)
    random_gauges!(U)
    hb = Heatbath(U, true, 10, 1, "subgroups", 4)
    for i in 1:N
        update!(hb, U)
        MetaQCD.is_special_unitary(U[1], prec=1e-6) || return i
    end
    return N
end

module BenchDirac

using BenchmarkTools
using LinearAlgebra
using MetaQCD

import Random

MetaQCD.MetaIO.set_global_logger!(1, nothing; tc=false) # INFO: disable logging during benchmarks

ops = (
    WilsonDiracOperator,
    # WilsonEOPreDiracOperator,
    StaggeredDiracOperator,
    StaggeredEOPreDiracOperator,
)

titles = (
    "Wilson",
    # "Wilson (Even-Odd preconditioned)",
    "Staggered",
    "Staggered (Even-Odd preconditioned)",
)

Random.seed!(1234)

N = 16

suite = BenchmarkGroup()

for (i, dirac) in enumerate(ops)
    s = suite["$(titles[i])"] = BenchmarkGroup()
    for T in (Float32, Float64)
        U = Gaugefield{CPU,T,WilsonGaugeAction}(N, N, N, N, 6.0)
        D = dirac(U, 0.01; csw=1.0)
        ϕ = Spinorfield(D.temp)
        ψ = Spinorfield(D.temp)

        random_gauges!(U)
        gaussian_pseudofermions!(ϕ)
        if dirac === StaggeredEOPreDiracOperator
            D_U = MetaQCD.DiracOperators.DdaggerD(D(U))
        else
            D_U = D(U)
        end

        s["$(T)"] = @benchmarkable(mul!($ψ, $D_U, $ϕ))
    end
end

end

BenchDirac.suite

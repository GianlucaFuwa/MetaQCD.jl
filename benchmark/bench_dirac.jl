module BenchDirac

using BenchmarkTools
using LinearAlgebra
using MetaQCD

import Random

MetaQCD.MetaIO.set_global_logger!(1, nothing; tc=false) # INFO: disable logging during benchmarks

ops = (
    WilsonDiracOperator,
    WilsonDiracOperator,
    # WilsonEOPreDiracOperator,
    StaggeredDiracOperator,
    # StaggeredEOPreDiracOperator,
)

titles = (
    "Wilson",
    "Wilson-Clover",
    # "Wilson (Even-Odd preconditioned)",
    "Staggered",
    # "Staggered (Even-Odd preconditioned)",
)

Random.seed!(1234)

N = 32
backend_str = "cpu"
backend = MetaQCD.Fields.BACKENDS[backend_str]

suite = BenchmarkGroup()

for (i, dirac) in enumerate(ops)
    s = suite["$(titles[i])"] = BenchmarkGroup()
    for T in (Float32, Float64)
        U = Gaugefield{backend,T,WilsonGaugeAction}(N, N, N, N, 6.0)
        csw = titles[i] == "Wilson-Clover" ? 1.0 : 0.0
        D = dirac(U, 0.01; csw=csw)
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

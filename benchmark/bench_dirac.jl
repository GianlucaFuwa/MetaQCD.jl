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
    # numprocs_cart = (1, 1, 1, 1)
    # halo_width = titles[i] == "Wilson-Clover" ? 2 : 1
    s = suite["$(titles[i])"] = BenchmarkGroup()
    for T in (Float16, Float32, Float64)
        U = Gaugefield{backend,T,WilsonGaugeAction,12}(N, N, N, N, 6.0)
        csw = titles[i] == "Wilson-Clover" ? 1.0 : 0.0
        D = dirac(U, 0.01; csw=csw)
        ϕ = similar(D.temp)
        ψ = similar(D.temp)

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

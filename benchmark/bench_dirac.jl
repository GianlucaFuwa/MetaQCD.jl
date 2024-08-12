module BenchDirac

using BenchmarkTools
using LinearAlgebra
using MetaQCD

import Random

ops = (
    WilsonDiracOperator,
    # WilsonEOPreDiracOperator,
    StaggeredDiracOperator,
    # StaggeredEOPreDiracOperator,
)

Random.seed!(1234)

N = 16

suite = BenchmarkGroup()

for dirac in ops
    s = suite["$(dirac)"] = BenchmarkGroup()
    for T in (Float32, Float64)
        U = Gaugefield{CPU,T,WilsonGaugeAction}(N, N, N, N, 6.0)
        D = dirac(U, 0.01; csw=1.0)
        ϕ = Fermionfield(D.temp)
        ψ = Fermionfield(D.temp)

        random_gauges!(U)
        gaussian_pseudofermions!(ϕ)
        D_U = D(U)

        s["$(T)"] = @benchmarkable(mul!($ψ, $D_U, $ϕ))
    end
end

end

BenchDirac.suite

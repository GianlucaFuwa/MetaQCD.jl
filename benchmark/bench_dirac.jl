module BenchDirac

using BenchmarkTools
using LinearAlgebra
using MetaQCD

import Random

ops = (
    WilsonDiracOperator,
    # WilsonEOPreDiracOperator,
    StaggeredDiracOperator,
    StaggeredEOPreDiracOperator,
)

Random.seed!(1234)

N = 16

suite = BenchmarkGroup()

for dirac in ops
    s = suite["$(dirac)"] = BenchmarkGroup()
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

using MetaQCD
using MetaQCD.Utils
using Aqua
using Random
using Test
using Polyester

include("test_meas.jl")
include("test_derivative.jl")
include("test_update.jl")
include("test_gflow.jl")
include("test_reversibility.jl")

@testset "MetaQCD.jl" begin
    plaq, poly, tc_plaq, tc_clover, tc_improved, wl_1x1 = SU3testmeas()
    @test isapprox(0.587818337847024, plaq)
    @test isapprox(0.5255246068616176 - 0.15140850971249734im, poly)
    @test isapprox(-0.2730960126400261, tc_plaq)
    @test isapprox(-0.027164585971545994, tc_clover)
    @test isapprox(-0.03210085960569041, tc_improved)
    @test isapprox(0.587818337847024, wl_1x1)
    # @test isapprox(0.26280892864238975, energy_density)
    relerrors = SU3testderivative()
    @test length(findall(x -> abs(x) > 1e-3, relerrors)) == 0

    @test SU3testgradflow()
end

if VERSION >= v"1.9"
    Aqua.test_all(MetaQCD, stale_deps=false)
end

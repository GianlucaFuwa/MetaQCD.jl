using MetaQCD
using Test

include("testmeas.jl")
include("testderivative.jl")
include("testupdate.jl")

@testset "MetaQCD.jl" begin
    plaq, poly, top_plaq, top_clover, top_improved, wilsonloop1x1 = SU3testmeas()
    @test isapprox(0.587818337847024, plaq)
    @test isapprox(0.5255246068616176 - 0.15140850971249734im, poly)
    @test isapprox(-0.2730960126400261, top_plaq)
    @test isapprox(-0.027164585971545994, top_clover)
    @test isapprox(-0.03210085960569041, top_improved)
    @test isapprox(0.587818337847024, wilsonloop1x1)

    relerrors = SU3testderivative()
    @test length(findall(x -> abs(x) > 1e-4, relerrors)) == 0

    accrate = SU3testupdate()
    @test accrate == 1
end
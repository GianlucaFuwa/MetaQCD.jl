using MetaQCD
using MetaQCD.Utils
using MetaQCD.Updates: calc_dQdU_bare!
using Aqua
using Random
using Test
using Polyester
using AMDGPU: ROCBackend
using CUDA

include("test_meas.jl")
include("test_derivative.jl")
include("test_update.jl")
include("test_gflow.jl")
include("test_clinalg.jl")
# include("test_reversibility.jl")

@testset "Linear Algebra Tests" begin
    SU3testlinalg()
end

@testset "CPU Tests" begin
    # CUDA.allowscalar(false)
    backend = nothing
    plaq, poly, tc_plaq, tc_clover, tc_improved, wl_1x1 = SU3testmeas(backend)
    @test isapprox(0.587818337847024, plaq)
    @test isapprox(0.5255246068616176 - 0.15140850971249734im, poly)
    @test isapprox(-0.2730960126400261, tc_plaq)
    @test isapprox(-0.027164585971545994, tc_clover)
    @test isapprox(-0.03210085960569041, tc_improved)
    @test isapprox(0.587818337847024, wl_1x1)
    relerrors = SU3testderivative(backend)
    @test length(findall(x -> abs(x) > 1e-4, relerrors)) == 0

    @test SU3testgradflow(backend)
    @test SU3testupdate(backend; update_method="heatbath")
    @test SU3testupdate(backend; update_method="metropolis", gaction=IwasakiGaugeAction)
    @test SU3testupdate(backend; update_method="hmc")
end

@testset "GPU Tests" begin
    CUDA.allowscalar(false)
    backend = CUDABackend
    plaq, poly, tc_plaq, tc_clover, tc_improved, wl_1x1 = SU3testmeas(backend)
    @test isapprox(0.587818337847024, plaq)
    @test isapprox(0.5255246068616176 - 0.15140850971249734im, poly)
    @test isapprox(-0.2730960126400261, tc_plaq)
    @test isapprox(-0.027164585971545994, tc_clover)
    @test isapprox(-0.03210085960569041, tc_improved)
    @test isapprox(0.587818337847024, wl_1x1)
    relerrors = CUDA.@allowscalar(SU3testderivative(backend))
    @test length(findall(x -> abs(x) > 1e-4, relerrors)) == 0

    @test SU3testgradflow(backend)
    @test SU3testupdate(backend; update_method="heatbath")
    @test SU3testupdate(backend; update_method="metropolis", gaction=IwasakiGaugeAction)
    @test SU3testupdate(backend; update_method="hmc")
end

if VERSION >= v"1.9"
    Aqua.test_all(MetaQCD; stale_deps=false, ambiguities=false)
end

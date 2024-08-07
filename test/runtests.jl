using MetaQCD
using MetaQCD.Utils
using MetaQCD.Updates: calc_dQdU_bare!
using Aqua
using Random
using Test
using Polyester
using CUDA

include("./test_io.jl")
include("./test_checkpoint.jl")
include("./test_meas.jl")
include("./test_derivative.jl")
include("./test_fderivative.jl")
include("./test_update.jl")
include("./test_gflow.jl")
include("./test_clinalg.jl")
# include("test_reversibility.jl")

# @testset "Linear Algebra Tests" begin
#     test_cdot()
#     test_ckron()
#     test_cmvmul()
#     # test_spin_color()
#     test_cmatmul()
# end
#
# @testset "IO Tests" begin
#     test_io()
#     test_checkpoint()
# end

@testset "CPU Tests" begin
    # CUDA.allowscalar(false)
    backend = CPU
    plaq, poly, tc_plaq, tc_clover, tc_improved, wl_1x1 = test_measurements(backend)
    @test isapprox(0.587818337847024, plaq)
    @test isapprox(0.5255246068616176 - 0.15140850971249734im, poly)
    @test isapprox(-0.2730960126400261, tc_plaq)
    @test isapprox(-0.027164585971545994, tc_clover)
    @test isapprox(-0.03210085960569041, tc_improved)
    @test isapprox(0.587818337847024, wl_1x1)

    relerrors = test_derivative(backend)
    @test length(findall(x -> abs(x) > 1e-4, relerrors[:, 2])) == 0
    relerrors = test_fderivative(
        backend; dirac="staggered", mass=0.01, single_flavor=true
    )
    @test length(findall(x -> abs(x) > 1e-4, relerrors[:, 2])) == 0
    relerrors = test_fderivative(
        backend; dirac="staggered", mass=0.01, single_flavor=true, eoprec=true
    )
    @test length(findall(x -> abs(x) > 1e-4, relerrors[:, 2])) == 0
    relerrors = test_fderivative(
        backend; dirac="wilson", mass=0.01, single_flavor=true
    )
    @test length(findall(x -> abs(x) > 1e-4, relerrors[:, 2])) == 0
    relerrors = test_fderivative(
        backend; dirac="wilson", mass=0.01, single_flavor=false, eoprec=true # TODO:Nf=1 test with eo wils
    )
    # @test length(findall(x -> abs(x) > 1e-4, relerrors[:, 2])) == 0

    @test test_gradflow(backend)
    @test test_update(backend; update_method="heatbath")
    @test test_update(backend; update_method="metropolis", gaction=IwasakiGaugeAction)
    @test test_update(backend; update_method="hmc")
end

if CUDA.functional(true)
# if false
    @testset "CUDA Tests" begin
        CUDA.allowscalar(false)
        backend = CUDABackend
        plaq, poly, tc_plaq, tc_clover, tc_improved, wl_1x1 = test_measurements(backend)
        @test isapprox(0.587818337847024, plaq)
        @test isapprox(0.5255246068616176 - 0.15140850971249734im, poly)
        @test isapprox(-0.2730960126400261, tc_plaq)
        @test isapprox(-0.027164585971545994, tc_clover)
        @test isapprox(-0.03210085960569041, tc_improved)
        @test isapprox(0.587818337847024, wl_1x1)

        relerrors = CUDA.@allowscalar(test_derivative(backend))
        @test length(findall(x -> abs(x) > 1e-4, relerrors[:, 2])) == 0
        relerrors = CUDA.@allowscalar(test_fderivative(
            backend; dirac="staggered", mass=0.01, single_flavor=true
        ))
        @test length(findall(x -> abs(x) > 1e-4, relerrors[:, 2])) == 0
        # relerrors = CUDA.@allowscalar(test_fderivative(
        #     backend; dirac="staggered", mass=0.01, single_flavor=true, eoprec=true
        # ))
        # @test length(findall(x -> abs(x) > 1e-4, relerrors[:, 2])) == 0
        relerrors = CUDA.@allowscalar(test_fderivative(
            backend; dirac="wilson", mass=0.01, single_flavor=true
        ))
        @test length(findall(x -> abs(x) > 1e-4, relerrors[:, 2])) == 0
        # relerrors = CUDA.@allowscalar(test_fderivative(
        #     backend; dirac="wilson", mass=0.01, single_flavor=false, eoprec=true
        # ))
        # @test length(findall(x -> abs(x) > 1e-4, relerrors[:, 2])) == 0

        @test test_gradflow(backend)
        @test test_update(backend; update_method="heatbath")
        @test test_update(backend; update_method="metropolis", gaction=IwasakiGaugeAction)
        @test test_update(backend; update_method="hmc")
    end
end

if VERSION >= v"1.9"
    Aqua.test_all(MetaQCD; stale_deps=false, ambiguities=false)
end

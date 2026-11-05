# MPI measurement/derivative regression entry point. CPU is the default;
# set METAQCD_TEST_BACKEND=rocm to exercise the GPU implementation.
pushfirst!(LOAD_PATH, dirname(@__DIR__))
using LinearAlgebra
using MetaQCD
using MetaQCD.Utils
using Random
using Test
const TEST_BACKEND = if get(ENV, "METAQCD_TEST_BACKEND", "cpu") == "rocm"
    @eval using AMDGPU
    ROCBackend
else
    CPU
end
include("test_meas.jl")
include("test_derivative.jl")
mpi_init()
@assert mpi_size() in (1, 4)
grid = mpi_size() == 4 ? (1, 2, 2, 1) : (1, 1, 1, 1)
part = get(ENV, "METAQCD_TEST_PART", "all")
@test part in ("all", "measurements", "derivatives")
@testset "MPI observables and derivatives ($TEST_BACKEND, $(mpi_size()) ranks)" begin
    if part != "derivatives"
        @testset "test_meas.jl" begin
            test_measurements(; backend=TEST_BACKEND, numprocs_cart=grid, halo_width=2)
        end
    end
    if part != "measurements"
        n = parse(Int, get(ENV, "METAQCD_DERIVATIVE_N", "16"))
        errors = if TEST_BACKEND == CPU
            test_derivative(; backend=TEST_BACKEND, numprocs_cart=grid, halo_width=2, N=n)
        else
            AMDGPU.allowscalar(true) # the finite-difference test reads its perturbed link
            try
                test_derivative(; backend=TEST_BACKEND, numprocs_cart=grid, halo_width=2, N=n)
            finally
                AMDGPU.allowscalar(false)
            end
        end
        @test all(isfinite, errors)
        @test maximum(abs, errors) < 1e-3
    end
end

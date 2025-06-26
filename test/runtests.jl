using MetaQCD
using MetaQCD.Utils
# using Aqua
using Random
using Test
using Polyester

include("./test_io.jl")
include("./test_checkpoint.jl")
include("./test_meas.jl")
include("./test_derivative.jl")
include("./test_fderivative.jl")
include("./test_update.jl")
include("./test_gflow.jl")
include("./test_clinalg.jl")
# include("test_reversibility.jl")

function runtests(; backend=CPU, nprocs_cart=(1, 1, 1, 1))
    @testset verbose = true "$backend Tests" begin
        mpi_size() != 1 && @level1("\nMPI Tests...")

        # INFO: Cant partition in time dimension if we want to measure polyakov loop
        if mpi_size() == 1
            test_measurements(; backend)
        elseif mpi_size() == 2
            test_measurements(; backend, nprocs_cart=(1, 1, 2, 1), halo_width=2)
        elseif mpi_size() == 4; 
            test_measurements(; backend, nprocs_cart=(1, 2, 2, 1), halo_width=2)
        else
            error("mpi_size has to be 1, 2 or 4 in tests")
        end

        # gauge derivative
        test_derivative(; backend, nprocs_cart, halo_width=2)

        # staggered derivative
        test_fderivative(; 
            backend,
            nprocs_cart,
            halo_width=1,
            dirac="staggered",
            mass=0.01,
            single_flavor=true,
        )

        # staggered-hoelbling1234 derivative
        test_fderivative(;
            backend,
            nprocs_cart,
            halo_width=2,
            dirac="staggered_h1234",
            mass=0.01,
            single_flavor=true,
        )

        # staggered-hoelbling1342 derivative
        test_fderivative(;
            backend,
            nprocs_cart,
            halo_width=2,
            dirac="staggered_h1342",
            mass=0.01,
            single_flavor=true,
        )

        # wilson derivative
        test_fderivative(;
            backend,
            nprocs_cart,
            halo_width=1,
            dirac="wilson",
            mass=0.01,
            single_flavor=true,
            csw=0,
        )

        # wilson-clover derivative
        test_fderivative(;
            backend,
            nprocs_cart,
            halo_width=2, # INFO: Halo width has to be 2 here
            dirac="wilson",
            mass=0.01,
            single_flavor=true,
            csw=1.78,
        )

        # staggered eo-pre derivative
        test_fderivative(;
            backend,
            nprocs_cart,
            halo_width=1,
            dirac="staggered",
            mass=0.01,
            single_flavor=true,
            eoprec=true,
        )

        # wilson eo-pre derivative
        test_fderivative(;
            backend,
            nprocs_cart,
            halo_width=1,
            dirac="wilson",
            mass=0.01,
            single_flavor=false,
            eoprec=true,
            csw=0,
        )

        # FIXME: wilson-clover eo-pre derivative
        # test_fderivative(;
        #     backend,
        #     nprocs_cart,
        #     halo_width=2, # INFO: Halo width has to be 2 here
        #     dirac="wilson",
        #     mass=0.01,
        #     single_flavor=false,
        #     eoprec=true,
        #     csw=1.78,
        # )

        test_gradflow(; backend, nprocs_cart, halo_width=1)

        if mpi_size() == 1 # INFO: Local updates only without distributed fields
            test_update(backend; update_method="heatbath")
            test_update(backend; update_method="metropolis", gaction=IwasakiGaugeAction)
        end

        test_update(backend; update_method="hmc", hmc_integrator="Leapfrog")
        test_update(backend; update_method="hmc", hmc_integrator="OMF2")
        test_update(backend; update_method="hmc", hmc_integrator="OMF4")

        # Run a short simulation as final test (doesnt work on github actions)
        # if backend == CPU
        #     if mpi_size() == 1 # INFO: Local updates only without distributed fields
        #         run_sim(joinpath(pkgdir(MetaQCD, "test", "parameters_test.toml")))
        #     elseif mpi_size() == 2
        #         run_sim(joinpath(pkgdir(MetaQCD, "test", "parameters_test_mpi.toml")))
        #     end
        # end
    end
end

sout = mpi_amroot() ? stdout : devnull
redirect_stdout(sout) do
    if mpi_size() == 1
        @testset "Linear Algebra Tests" begin
            test_cdot()
            test_ckron()
            test_cmvmul()
            # test_spin_color() # FIXME: Fix these tests
            test_cmatmul()
        end

        @testset "IO Tests" begin
            test_io()
            # test_checkpoint()
        end
    end

    runtests(; nprocs_cart=(1, 1, 1, mpi_size()))
end

# using AMDGPU, AMDGPU: allowscalar
# using CUDA, CUDA: allowscalar

# runtests(; backend=ROCBackend)

# if mpi_amroot() && mpi_size() == 1
#     if VERSION >= v"1.9"
#         Aqua.test_all(MetaQCD; stale_deps=false, ambiguities=false)
#     end
# end

# if mpi_size() == 1
#     cmd = Base.julia_cmd()
#     path = joinpath(@__DIR__, "runtests.jl")
#     run(`$(Utils.MPI.mpiexec()) -n 2 $(cmd) --project --startup-file=no $(path)`)
# end

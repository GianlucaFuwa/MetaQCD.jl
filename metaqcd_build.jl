using Pkg
Pkg.activate(@__DIR__(); io=devnull)

using MetaQCD.Utils
using MetaQCD: build_bias
using MetaQCD.Parameters: lower_case

COMM = mpi_init()
# @assert mpi_size() == 1 "metaqcd_sim can not be used with mpi, only metaqcd_build"

if length(ARGS) == 0 && mpi_myrank() == 0
    error("""
    A parameter file has to be given as the first input:
    julia metaqcd_build.jl parameters.toml
    """)
end

backend = "cpu"

if length(ARGS) == 2
    mpi_amroot() && @info(
        "If you get prompted to install a package here, make sure you do it in your" *
        "GLOBAL julia environment, i.e., not under a project environment"
    )
    if lower_case(ARGS[2]) == "cpu"
    elseif lower_case(ARGS[2]) == "cuda"
        using CUDA
        backend = "cuda"
    elseif lower_case(ARGS[2]) == "rocm"
        using AMDGPU
        backend = "rocm"
    else
        mpi_myrank() == 0 && error("""
              When a second input is given, it has to specify the backend to be used, so the package can be loaded.
              Note, that the backend also has to be set in the parameter file 
              Supported backends are:
              - cpu
              - cuda
              - rocm
              Your input was \"$(ARGS[2])\"
              """)
    end
end

if mpi_parallel() && mpi_myrank() == 0
    println("$(mpi_size()) mpi processes will be used")
end

mpi_barrier()

build_bias(ARGS[1]; backend=backend, mpi_multi_sim=with_mpi)

using Pkg
Pkg.activate(@__DIR__(); io=devnull)

using MetaQCD.MPI
using MetaQCD: run_sim
using MetaQCD.Parameters: lower_case

@assert MPI.Comm_size(MPI.COMM_WORLD) == 1 "metaqcd_sim can not be used with mpi, only metaqcd_build"

if length(ARGS) == 0
    error("""
    A parameter file has to be given as the first input:
    julia metaqcd_sim.jl parameters.toml
    """)
end

backend = "cpu"

if length(ARGS) == 2
    @info(
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
        error("""
              When a second input is given, it has to specify the backend to be used, so the package can be loaded.
              Note, that the backend also has to be set in the parameter file.
              Supported backends are:
              - cpu
              - cuda
              - rocm
              Your input was \"$(ARGS[2])\"
              """)
    end
end

with_mpi = MPI.Comm_size(COMM) > 1
if with_mpi && MYRANK == 0
    println("$(MPI.Comm_size(COMM)) streams will be used")
end

MPI.Barrier(COMM)

run_sim(ARGS[1]; backend=backend, mpi_enabled=with_mpi)

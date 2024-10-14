using Pkg
Pkg.activate(@__DIR__(); io=devnull)

using MetaQCD.Utils
using MetaQCD: @level1, run_sim
using MetaQCD.Parameters: lower_case

@assert length(ARGS) != 0 """
A parameter file has to be given as the first input:
julia metaqcd_sim.jl parameters.toml
"""

backend = "cpu"

if length(ARGS) == 2
    @level1 """
    If you get prompted to install a package here, make sure you do it in your \
    GLOBAL julia environment, i.e., not under a project environment
    """

    if lower_case(ARGS[2]) == "cpu"
    elseif lower_case(ARGS[2]) == "cuda"
        using CUDA
        backend = "cuda"
    elseif lower_case(ARGS[2]) ∈ ("rocm", "amd")
        using AMDGPU
        backend = "rocm"
    else
        throw(ArgumentError(
            """
            When a second input is given, it has to specify the backend to be used, \
            so the package can be loaded.
            Note, that the backend also has to be set in the parameter file.
            Supported backends are:
            - cpu
            - cuda
            - rocm
            Your input was \"$(ARGS[2])\"
            """
        ))
    end
end

mpi_parallel() && @level1("[ $(mpi_size()) MPI processes are being used")

run_sim(ARGS[1]; backend=backend)

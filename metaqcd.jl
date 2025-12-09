# using Pkg
# Pkg.activate(@__DIR__(); io=devnull)
#
using MetaQCD.Utils
using MetaQCD: @level1, run_build, run_sim, construct_params_from_toml

mpi_amroot() && print_startup()

function parse_args(args)
    parameterfile = args[end]
    @assert length(args) >= 1 && isfile(parameterfile) """
    An existing parameter file has to be given as an input, e.g.:
    julia metaqcd.jl parameters.toml
    You either did not provide a file or the file you provided does not exist.
    """
    parameters = construct_params_from_toml(parameterfile)
    return parameters, parameters.mode, parameters.backend
end

parameters, mode, backend = parse_args(ARGS)

@level1("##### Mode: $(mode)")
@level1("##### Number of ranks: $(mpi_size())")
@level1("##### Number of threads on each rank: $(Base.Threads.nthreads())")
@level1("##### Working Directory: $(pwd())\n")
# @level1("##### MetaQCD.jl version: $(METAQCD_VERSION)\n")

if backend != "cpu"
    @level1 """
    If you get prompted to install a package here, make sure you do it in your \
    GLOBAL julia environment, i.e., not under a project environment
    """

    if backend == "cuda"
        using CUDA
    elseif backend ∈ ("roc", "rocm", "amd", "amdgpu")
        using AMDGPU
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

            Your input was \"$(backend)\"
            """
        ))
    end
end

if mode == "sim"
    run_sim(parameters)
elseif mode == "build"
    run_build(parameters)
else
    throw(ArgumentError(
        """
        The supplied \"mode\" in the parameter file is invalid. The two options are:

        \"sim\"   to run a simulation with or without Metadynamics or
        \"build\" for building a bias potential with possibly multiple walkers

        Your input was \"$(mode)\"
        """
    ))
end

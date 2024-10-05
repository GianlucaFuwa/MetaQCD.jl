using Pkg
Pkg.activate(@__DIR__(); io=devnull)

using MetaQCD.Utils
using MetaQCD: @level1, build_bias, run_sim

function parse_args(args)
    parameterfile = args[1]
    @assert length(args) > 2 && isfile(parameterfile) """
    An existing parameter file has to be given as the first input, e.g.:
    julia metaqcd.jl parameters.toml -mode=sim

    You either did not provide a file or the file you provided does not exist.
    """

    @assert count(x -> occursin("-mode", x), args) == 1 """
    The flag \"-mode\" has to be set after the parameter file.
    Options are:

    \"-mode=sim\"   to run a simulation with or without Metadynamics or
    \"-mode=build\" for building a bias potential with possibly multiple walkers
    """

    mode = split(args[findfirst(x -> occursin("-mode", x), args)], "=")[2]
    backend = try
        split(args[findfirst(x -> occursin("-backend", x), args)], "=")[2]
    catch _
        "cpu"
    end

    return parameterfile, mode, backend
end

parameterfile, mode, backend = parse_args(ARGS)

@level1 "[ Mode: $(mode)\n"

if backend != "cpu"
    @level1 """
    If you get prompted to install a package here, make sure you do it in your \
    GLOBAL julia environment, i.e., not under a project environment
    """

    if backend == "cuda"
        using CUDA
    elseif backend ∈ ("rocm", "amd")
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

mpi_parallel() && @level1("[ $(mpi_size()) MPI processes are being used")

if mode == "sim"
    run_sim(parameterfile; backend=backend)
elseif mode == "build"
    build_bias(parameterfile; backend=backend, mpi_multi_sim=with_mpi)
else
    throw(ArgumentError(
        """
        The supplied \"-mode\" is invalid. The two options are:

        \"-mode=sim\"   to run a simulation with or without Metadynamics or
        \"-mode=build\" for building a bias potential with possibly multiple walkers

        Your input was \"$(mode)\"
        """
    ))
end

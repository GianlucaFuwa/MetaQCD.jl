using MPI
include("./src/MetaQCD.jl")
using .MetaQCD

function mpirun()
    run_build(ARGS[1]; MPIparallel = true)
end

mpirun()

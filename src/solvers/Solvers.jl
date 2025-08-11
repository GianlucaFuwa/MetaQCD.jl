module Solvers

using Accessors
using LinearAlgebra
using StaticArrays
using ..Utils
using ..MetaIO

export SolverInfo, bicg!, bicg_stab!, cg!, cgnr!, mscg!, get_info

struct SolverInfo{F,S}
    solver::F
    tol::Float64
    maxiters::Int64
    datafile::S
end 

# TODO: Temporary for now
@inline function get_info(solver::SolverInfo)
    return solver.tol, solver.maxiters, solver.datafile
end

@inline function print_solverdata(datafile, iters, res)
    if datafile != ""
        set_ext!(datafile, MPI_INSTANCE[])
        fp = fopen(datafile, "a")
        printf(fp, "%-11i", iters)
        printf(fp, "%-25.15E", res)
        printf(fp, "%s", "action")
        newline(fp)
        fclose(fp)
    end

    return nothing
end

include("cg.jl")

end

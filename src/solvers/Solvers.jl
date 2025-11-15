module Solvers

using Accessors
using LinearAlgebra
using StaticArrays
using ..Logs
using ..Utils

export SolverInfo, get_info
export bicg!, bicg_stab!, cg!, cgnr!, mscg!, cg_mixed!, mscg_mixed!

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
        newline(fp)
        fclose(fp)
    end

    return nothing
end

@inline function print_solverdata(datafile, iters, res, elapsed_time)
    if datafile != ""
        set_ext!(datafile, MPI_INSTANCE[])
        fp = fopen(datafile, "a")
        printf(fp, "%-11i", iters)
        printf(fp, "%-25.15E", res)
        printf(fp, "%-25.6E", elapsed_time)
        newline(fp)
        fclose(fp)
    end

    return nothing
end

@inline function print_solverdata(datafile, outer_iters, inner_iters, res, elapsed_time)
    if datafile != ""
        set_ext!(datafile, MPI_INSTANCE[])
        fp = fopen(datafile, "a")
        printf(fp, "%-11i", outer_iters)
        printf(fp, "%-11i", inner_iters)
        printf(fp, "%-25.15E", res)
        printf(fp, "%-25.6E", elapsed_time)
        newline(fp)
        fclose(fp)
    end

    return nothing
end

include("cg.jl")
include("cg_mixed.jl")

end

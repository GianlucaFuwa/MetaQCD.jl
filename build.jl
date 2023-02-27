using Distributed
@everywhere include("./src/MetaQCD.jl")
@everywhere using .MetaQCD
@everywhere using Random
@everywhere using Printf
@everywhere using DelimitedFiles

if length(ARGS) == 0
    error("""
    Use input file:
    Like,
    julia build.jl parameters.jl
    """)
end

function runtest()
    run_build(ARGS[1])
end

@time runtest()
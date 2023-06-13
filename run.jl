include("./src/MetaQCD.jl")
using .MetaQCD

if length(ARGS) == 0
    error("""
    Use input file, like:
    julia run.jl parameters.toml
    """
    )
end

function runtest()
    run_sim(ARGS[1])
end

@time runtest()

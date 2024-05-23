include("./src/MetaQCD.jl")
using .MetaQCD: run_sim

if length(ARGS) == 0
    error("""
    Use input file, like:
    julia run.jl parameters.toml
    """
    )
end

function run()
    run_sim(ARGS[1])
end

@time run()

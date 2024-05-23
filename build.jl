include("./src/MetaQCD.jl")
using .MetaQCD: run_build

if length(ARGS) == 0
    error("""
    Use input file, like:
    julia run.jl parameters.toml
    """
    )
end

function build()
    run_build(ARGS[1])
end

@time build()

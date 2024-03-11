pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
Pkg.activate(@__DIR__)
Pkg.instantiate()
using Documenter
using MetaQCD

makedocs(
    sitename = "MetaQCD.jl",
    authors = "Gianluca Fuwa and Contributors",
    clean = true,
    doctest = false,
    modules = [MetaQCD],
    pages = [
        "MetaQCD.jl: Metadynamics in Lattice Yang-Mills" => "index.md",
        "Usage" => "usage.md",
        "Creating Gaugefields" => "gaugefields.md",
        "Updating a Gaugefield" => "updates.md",
        "Biased Sampling Methods" => "biased_sampling.md",
        "Visualization" => "viz.md",
        "Parameters" => "parameters.md",
        "Utility Functions" => "utils.md",
        # "Metadynamics" => "metadynamics.md",
    ],
    checkdocs = :none,
    warnonly = [:cross_references],
    format = Documenter.HTML(),
)

deploydocs(repo = "github.com/GianlucaFuwa/MetaQCD.jl.git")
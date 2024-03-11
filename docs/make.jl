push!(LOAD_PATH, "../src/")
using Documenter
using MetaQCD

makedocs(sitename="My Docs")

deploydocs(repo = "github.com/GianlucaFuwa/MetaQCD.jl.git")
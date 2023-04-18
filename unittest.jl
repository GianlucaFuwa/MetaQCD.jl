include("./src/system/system_parameters.jl")
include("./src/system/utils.jl")
include("./src/system/verbose.jl")
include("./src/fields/gaugefields.jl")
include("./src/measurements/AbstractMeasurement.jl")
using BenchmarkTools
using StaticArrays
using Random
using .Utils
import ..Gaugefields: Gaugefield,IdentityGauges,RandomGauges
import ..AbstractMeasurement_module

rng = Xoshiro(1206)

g = IdentityGauges(4,4,4,4,5.0)



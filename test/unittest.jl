using Distributed

@everywhere include("src/MetaQCD.jl")
@everywhere using .MetaQCD
@everywhere using Random 
@everywhere using StaticArrays
@everywhere using LinearAlgebra
@everywhere using BenchmarkTools

rng = Xoshiro(1209)
origin = SA[1,1,1,1]
# Test local action difference:
gfield = Gaugefield(16,16,16,16,1.0)
RandomGauges!(gfield,rng)

#@btime calc_staplesum(gfield,1,origin)

#=
X = gen_proposal(rng,0.1)
ds = loc_action_diff(gfield,1,origin,X)
gfield.U[:,:,1,CartesianIndex(Tuple(origin))] = X*gfield.U[:,:,1,CartesianIndex(Tuple(origin))]
Sold = gfield.Sg
recalc_Sg!(gfield)
println("Error = $(abs(ds+(Sold-gfield.Sg)))")

loc_metro_sweep!(gfield,rng,0.1)
Salg = gfield.Sg
recalc_Sg!(gfield)
Sexact = gfield.Sg
println("Error = $(abs(Sexact-Salg))")

pfield = Liefield(8,8,8,8)
@time HMC!(gfield,pfield,0.0002,50,rng)
Shmc = gfield.Sg
println("HMC-diff = $(abs(Shmc-Sexact))")
=#
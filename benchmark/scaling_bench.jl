using MetaQCD, MetaQCD.Utils, MPI, LinearAlgebra, Chairmarks
using Random, Statistics, Printf

const FLOPS = Dict(
    "Staggered" => 587,
    "Wilson" => 1368, "Wilson-Clover" => 1368 + 1728
)

function distribute_procs(global_dims, numprocs)
    procsleft = numprocs
    numprocs_cart = [1, 1, 1, 1]
    dim = 4
    while procsleft > 1
        dim = mod1(dim, 4)
        numprocs_cart[dim] *= 2
        @assert global_dims[dim] / numprocs_cart[dim] > 4 "too many proc
        s"
        procsleft /= 2
        dim -= 1
    end
    return (numprocs_cart...,)
end

B = CPU
T = Float64
GA = WilsonGaugeAction
global_dims = (32, 32, 32, 32)
numprocs_cart = distribute_procs(global_dims, mpi_size())
halo_width = 1

function mem_per_site(op, ::Type{T}) where T
    # 1 read fermion on site
    # 8 reads fermion neighbors in all μ-directions forward and backward
    # 1 write fermion on site
    # 8 reads gauge neighbors in all μ-directions forward and backward
    if op == "Staggered"
        return  2 * sizeof(T) * (10 * 3 + 8 * 9)
    elseif op == "Wilson"
        return  2 * sizeof(T) * (10 * 12 + 8 * 9)
    elseif op == "Wilson-Clover"
        # 1 read fermion on site
        # 1 write fermion on site
        # 6 x 16 read gauge for clover
        return  2 * sizeof(T) * (2 * 12 + 6 * (16 * 9))
    else
        error()
    end
end

result_dir = joinpath(@__DIR__, "scaling_results")
bstring = lowercase(string(B)); tstring = lowercase(string(T))
nstring = "$(global_dims[1])x$(global_dims[2])x$(global_dims[3])x$(global_dims[4])"
filename = "$(prod(numprocs_cart))procs_$(nstring)_$(bstring)_$(tstring).txt" 

U = Gaugefield{B,T,GA}(global_dims..., 6.0; numprocs_cart, halo_width)

ϕ_stagg = Spinorfield(U; staggered=true); ψ_stagg = Spinorfield(U; staggered=true);
D_stagg = StaggeredDiracOperator(U, 0.01);

ϕ_wils = Spinorfield(U); ψ_wils = Spinorfield(U);
D_wils = WilsonDiracOperator(U, 0.01);
D_wils_clov = WilsonDiracOperator(U, 0.01; csw=1.78);

random_gauges!(U)
gaussian_pseudofermions!(ϕ_stagg)
gaussian_pseudofermions!(ϕ_wils)

mul!(ψ_stagg, (D_stagg(U)), ϕ_stagg)
mul!(ψ_wils, (D_wils(U)), ϕ_wils)
# mul!(ψ_wils, (D_wils_clov(U)), ϕ_wils)

# mpi_barrier()
# println("Benchmark Halo Exchange ($(mpi_myrank()))")
# _b = @be _ update_halo!($U) mpi_barrier() evals=1 samples=100 seconds=5
# update_halo!(U)
# io = IOBuffer()
# show(io, "text/plain", _b)
# s = String(take!(io))
# mpi_amroot() && println(s)
if mpi_amroot()
    fp = open(joinpath(result_dir, filename), "w+")
    println(
        fp,
        rpad("Op", 20),
        rpad("medianperf", 20),
        rpad("meanperf", 20),
        rpad("stdperf", 20),
        rpad("medianbandw", 20),
        rpad("meanbandw", 20),
        rpad("stdbandw", 20),
    )
end

mpi_barrier()
println("Benchmark Staggered Operator ($(mpi_myrank()))")
_b = @be _ mul!($ψ_stagg, $(D_stagg(U)), $ϕ_stagg) mpi_barrier() evals=1 samples=100 seconds=5
mem = mem_per_site("Staggered", T)
flops = FLOPS["Staggered"]
perf = prod(global_dims)*flops/1e9; bw = prod(global_dims)*mem/1e9
m = mean(s.time for s in _b.samples); mperf = perf / m; mbw = bw / m
md = median(s.time for s in _b.samples); mdperf = perf / md; mdbw = bw / md
st = std(s.time for s in _b.samples); stperf = mperf - (perf / (m+st)); stbw = mbw - (bw / (m+st))
if mpi_amroot()
    str = @sprintf(
        "%-20s%-20.5f%-20.5f%-20.5f%-20.5f%-20.5f%-20.5f",
        "staggered", mdperf, mperf, stperf, mdbw, mbw, stbw
    )
    println(fp, str)
end

mpi_barrier()
println("Benchmark Wilson Operator ($(mpi_myrank()))")
_b = @be _ mul!($ψ_wils, $(D_wils(U)), $ϕ_wils) mpi_barrier() evals=1 samples=100 seconds=5
mem = mem_per_site("Wilson", T)
flops = FLOPS["Wilson"]
perf = prod(global_dims)*flops/1e9; bw = prod(global_dims)*mem/1e9
m = mean(s.time for s in _b.samples); mperf = perf / m; mbw = bw / m
md = median(s.time for s in _b.samples); mdperf = perf / md; mdbw = bw / md
st = std(s.time for s in _b.samples); stperf = mperf - (perf / (m+st)); stbw = mbw - (bw / (m+st))
if mpi_amroot()
    str = @sprintf(
        "%-20s%-20.5f%-20.5f%-20.5f%-20.5f%-20.5f%-20.5f",
        "wilson", mdperf, mperf, stperf, mdbw, mbw, stbw
    )
    println(fp, str)
end
#
# mpi_barrier()
# println("Benchmark Wilson-Clover Operator ($(mpi_myrank()))")
# _b = @be _ mul!($ψ_wils, $(D_wils_clov(U)), $ϕ_wils) mpi_barrier() evals=1 samples=100 seconds=5
# mem = mem_per_site("Wilson-Clover", T)
# flops = FLOPS["Wilson-Clover"]
# perf = prod(global_dims)*flops/1e9; bw = prod(global_dims)*mem/1e9
# m = mean(s.time for s in _b.samples); mperf = perf / m; mbw = bw / m
# md = median(s.time for s in _b.samples); mdperf = perf / md; mdbw = bw / md
# st = std(s.time for s in _b.samples); stperf = mperf - (perf / (m+st)); stbw = mbw - (bw / (m+st))
# if mpi_amroot()
#     str = @sprintf(
#         "%-20s%-20.5f%-20.5f%-20.5f%-20.5f%-20.5f%-20.5f",
#         "wilson-clover", mdperf, mperf, stperf, mdbw, mbw, stbw
#     )
#     println(fp, str)
# end

mpi_barrier()
mpi_amroot() && close(fp)

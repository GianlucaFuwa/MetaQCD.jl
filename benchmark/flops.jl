using Statistics

const FLOPS = Dict("Staggered" => 587, "Wilson" => 1368, "Wilson-Clover" => 1368 + 1728)
function  mem_per_site(op, ::Type{T}) where T
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

suite = include("bench_dirac.jl")
using .BenchDirac
tune!(suite)
results = run(suite)

for (op, flops) in FLOPS
    N = BenchDirac.N
    fp = open("benchmark/results_$(op)_$(N).txt", "w+")
    println(fp, "==== $(op) ====")

    for T in (Float64, Float32)
        mem = mem_per_site(op, T)
        println(fp, "$T:")
        println(fp, "   L = $(N)^4:")
        mintime = minimum(results.data[op]["$T"].times)
        avgtime = mean(results.data[op]["$T"].times)
        println(fp, "   Max: $(N^4*flops / mintime) GFLOPS")
        println(fp, "   Avg: $(N^4*flops / avgtime) GFLOPS")
        println()
        println(fp, "   Max: $(N^4*mem / mintime) GB/s")
        println(fp, "   Avg: $(N^4*mem / avgtime) GB/s")
    end

    close(fp)
end

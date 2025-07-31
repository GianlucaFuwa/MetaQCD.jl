using Statistics, MetaQCD.Utils, BenchmarkTools

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

if mpi_amroot()
    backend = BenchDirac.backend_str
    fp = open("benchmark/results_$(backend)_$(mpi_size())procs.txt", "w+")
    for (op, flops) in FLOPS
        N = BenchDirac.N
        println(fp, "==== $(op) (V = $N^4) ====")
        for T in (Float64, Float32, Float16)
            mem = mem_per_site(op, T)
            println(fp, "$(string(T)):\t$(rpad("Max", 9))$(rpad("Avg", 9))")
            mintime = minimum(results.data[op]["$T"].times)
            avgtime = mean(results.data[op]["$T"].times)
            maxflops = N^4*flops / mintime
            avgflops = N^4*flops / avgtime
            print(fp, "\t\t$(rpad(round(maxflops, digits=3), 9))")
            print(fp, "$(rpad(round(avgflops, digits=3), 9))")
            println(fp,  "   GFLOPs")
            maxmem = N^4*mem / mintime
            avgmem = N^4*mem / avgtime
            print(fp,"\t\t$(rpad(round(maxmem, digits=3), 9))")
            print(fp,"$(rpad(round(avgmem, digits=3), 9))")
            println(fp,  "   GB/s")
        end
    end
    close(fp)
end

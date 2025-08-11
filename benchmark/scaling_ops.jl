using MetaQCD, MetaQCD.Utils, MPI, LinearAlgebra, Chairmarks
using Random, Statistics, Printf
using AMDGPU

const FLOPS = Dict(
    "Copy" => 0,
    "Staggered" => 587,
    "Wilson" => 1368,
    "Wilson-Clover" => 1368 + 1728
)

const OPERATORS = (
    nothing,
    StaggeredDiracOperator,
    WilsonDiracOperator,
    WilsonDiracOperator,
)

const NAMES = (
    "Copy",
    "Staggered",
    "Wilson",
    "Wilson-Clover",
)

function bench_mul!(::Type{B}, ϕ, D, ψ) where B
    mul!(ψ, D, ϕ)
    MetaQCD.Fields.synchronize(B())
    return nothing
end

function bench_copy!(::Type{B}, a, b) where B
    copy!(a, b)
    MetaQCD.Fields.synchronize(B())
    return nothing
end

function main()
    B = ROCBackend
    GA = WilsonGaugeAction
    global_dims = (32, 32, 32, 32)
    numprocs_cart = distribute_procs(global_dims, mpi_size())
    halo_width = 1
    result_dir = joinpath(@__DIR__, "scaling_results")
    bstring = lowercase(string(B))
    filename = "$(prod(numprocs_cart))procs_$(bstring).txt" 

    if mpi_amroot()
        fp = open(joinpath(result_dir, filename), "w+")
        println(
            fp,
            rpad("", 15),
            rpad("max", 12),
            rpad("median", 12),
            rpad("mean", 12),
            rpad("std", 10),
        )
    end

    # for irank in 0:mpi_size()-1
    #     if mpi_myrank() == irank
    #         println("$(mpi_myrank()): device $(AMDGPU.device_id())")
    #         mpi_barrier()
    #     end
    # end

    for (i, operator) in enumerate(OPERATORS)
        opname = NAMES[i]
        for T in (Float16, Float32, Float64)
            tstring = lowercase(string(T))
            U = Gaugefield{B,T,GA}(global_dims..., 6.0; numprocs_cart, halo_width)
            if opname == "Copy"
                a = similar(U)
                b = similar(U)
                # a = Spinorfield(U)
                # b = Spinorfield(U)

                copy!(a, b) # warmup
                mpi_barrier()

                println("Benchmarking $(opname) $(tstring) ($(mpi_myrank()))")
                bench = @be _ $bench_copy!($B, $a, $b) mpi_barrier() evals=1 samples=100 seconds=5
            else
                staggered = opname == "Staggered"
                csw = opname == "Wilson-Clover" ? 1.78 : 0.0
                ϕ = Spinorfield(U; staggered)
                ψ = Spinorfield(U; staggered)
                D = operator(U, 0.01; csw=csw)

                random_gauges!(U)
                gaussian_pseudofermions!(ϕ)

                mul!(ψ, (D(U)), ϕ) # warmup

                mpi_barrier()

                println("Benchmarking $(opname) $(tstring) Operator ($(mpi_myrank()))")
                bench = @be _ $bench_mul!($B, $ψ, $(D)($U), $ϕ) mpi_barrier() evals=1 samples=100 seconds=5
            end

            flops = prod(global_dims) * FLOPS[opname] / 1e9
            mem = prod(global_dims) * mem_per_site(opname, T) / 1e9

            mintime = minimum(s.time for s in bench.samples)
            mediantime = median(s.time for s in bench.samples)
            meantime = mean(s.time for s in bench.samples)
            stdtime = std(s.time for s in bench.samples)

            maxperf = flops / mintime
            medianperf = flops / mediantime
            meanperf = flops / meantime
            stdperf = meanperf - (flops / (meantime+stdtime))

            maxbw = mem / mintime
            medianbw = mem / mediantime
            meanbw = mem / meantime
            stdbw = meanbw - (mem / (meantime+stdtime))

            if mpi_amroot()
                println(fp, "$(opname) $(tstring)")
                str = @sprintf(
                    "%-15s%-12.3f%-12.3f%-12.3f%-10.3f[GFLOPs]",
                    "", maxperf, medianperf, meanperf, stdperf
                )
                println(fp, str)
                str = @sprintf(
                    "%-15s%-12.3f%-12.3f%-12.3f%-10.3f[GB/s]",
                    "", maxbw, medianbw, meanbw, stdbw
                )
                println(fp, str)
            end

            mpi_barrier()
        end
    end

    mpi_amroot() && close(fp)
    mpi_barrier()
    return nothing
end

function distribute_procs(global_dims, numprocs)
    procsleft = numprocs
    numprocs_cart = [1, 1, 1, 1]
    dim = 4
    while procsleft > 1
        numprocs_cart[dim] *= 2
        @assert global_dims[dim] / numprocs_cart[dim] > 4 "too many procs"
        procsleft /= 2
        dim = dim == 2 ? 4 : dim - 1
    end
    return (numprocs_cart...,)
end

function mem_per_site(op, ::Type{T}) where T
    # 1 read fermion on site
    # 8 reads fermion neighbors in all μ-directions forward and backward
    # 1 write fermion on site
    # 8 reads gauge neighbors in all μ-directions forward and backward
    if op == "Staggered"
        return sizeof(Complex{T}) * (10 * 3 + 8 * 9)
    elseif op == "Wilson"
        return sizeof(Complex{T}) * (10 * 12 + 8 * 9)
    elseif op == "Wilson-Clover"
        # 1 read fermion on site
        # 1 write fermion on site
        # 6 x 16 read gauge for clover
        return sizeof(Complex{T}) * (2 * 12 + 6 * (16 * 9))
    elseif op == "Copy"
        return sizeof(Complex{T}) * (2 * 4 * 9)
        # return sizeof(Complex{T}) * (2 * 12)
    else
        error()
    end
end

main()


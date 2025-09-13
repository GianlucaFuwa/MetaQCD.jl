using MetaQCD, MetaQCD.Utils, MPI, LinearAlgebra, Chairmarks
using Random, Statistics, Printf
using AMDGPU
using MetaQCD.DiracOperators: solve_dirac!
using MetaQCD.Fields: update_halo!

const FLOPS = Dict(
    "Copy" => 0,
    "Dot-Staggered" => 22,
    "Dot-Wilson" => 94,
    "Staggered" => 587,
    "Wilson" => 1368,
    "Wilson-Clover" => 1368 + 1728,
    "Invert-Staggered" => 587 + 2*22 + 2*12 + 18, # op + 2dot + 2axpy + axpby
    "Invert-Wilson" => 1368 + 2*22 + 2*12 + 18, # op + 2dot + 2axpy + axpby
    "Halo-Exchange" => 0,
)

function bench_mul!(::Type{B}, ϕ, D, ψ, nlaunches) where B
    for _ in 1:nlaunches
        mul!(ψ, D, ϕ)
    end
    MetaQCD.Fields.synchronize(B())
    return nothing
end

function bench_copy!(::Type{B}, a, b, nlaunches) where B
    for _ in 1:nlaunches
        copy!(a, b)
    end
    MetaQCD.Fields.synchronize(B())
    return nothing
end

function bench_dot(::Type{B}, a, b, nlaunches) where B
    for _ in 1:nlaunches
        dot(a, b)
    end
    MetaQCD.Fields.synchronize(B())
    return nothing
end

function bench_exchange(::Type{B}, u, nlaunches) where B
    for _ in 1:nlaunches
        update_halo!((u,))
    end
    MetaQCD.Fields.synchronize(B())
    return nothing
end

function main()
    MetaQCD.MetaIO.set_global_logger!(4, nothing)
    nlaunches = 10
    B = ROCBackend
    GA = WilsonGaugeAction
    N = 12
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

    deviceid_printed = false

    for opname in keys(FLOPS)
        for T in (Float16, Float32, Float64)
            tstring = lowercase(string(T))
            U = Gaugefield{B,T,GA,N}(global_dims..., 6.0; numprocs_cart, halo_width)

            if !deviceid_printed
                for irank in 0:mpi_size()-1
                    if mpi_myrank() == irank
                        println("$(mpi_myrank()): device $(AMDGPU.device_id())")
                        mpi_barrier()
                    end
                end
                deviceid_printed = true
            end

            if opname == "Copy"
                a = similar(U)
                b = similar(U)
                # warmup
                copy!(a, b)
                mpi_barrier()
                println("Benchmarking $(opname) $(tstring) ($(mpi_myrank()))")
                bench = @be _ $bench_copy!($B, $a, $b, $nlaunches) mpi_barrier() evals=1 samples=100 seconds=5
            elseif opname == "Halo-Exchange"
                f = Spinorfield(U; staggered=true)
                # warmup
                update_halo!((f,))
                mpi_barrier()
                println("Benchmarking $(opname) $(tstring) ($(mpi_myrank()))")
                bench = @be _ $bench_exchange($B, $f, $nlaunches) mpi_barrier() evals=1 samples=100 seconds=5
            elseif contains(opname, "Dot")
                staggered = contains(opname, "Staggered")
                a = Spinorfield(U; staggered)
                b = Spinorfield(U; staggered)
                # warmup
                dot(a, b)
                mpi_barrier()
                println("Benchmarking $(opname) $(tstring) ($(mpi_myrank()))")
                bench = @be _ $bench_dot($B, $a, $b, $nlaunches) mpi_barrier() evals=1 samples=100 seconds=5
            elseif contains(opname, "Invert")
                T == Float64 || continue
                operator, staggered = if contains(opname, "Staggered")
                    StaggeredDiracOperator, true
                else
                    WilsonDiracOperator, false
                end
                D = operator(U, 0.01; csw=1.78)
                ϕ = Spinorfield(U; staggered)
                ψ = Spinorfield(U; staggered)
                random_gauges!(U)
                g = GradientFlow(U; integrator="euler", numflow=6, steps=1, tf=0.12)
                copy!(g.Uflow, U)
                for _ in 1:g.numflow
                    flow!(g)
                end
                copy!(U, g.Uflow)
                @show plaquette_trace_sum(g.Uflow)
                @show plaquette_trace_sum(U)
                gaussian_pseudofermions!(ϕ)
                temp1, temp2, temp3 = ntuple(_ -> Spinorfield(U; staggered), Val(3))
                tol = 1e-4
                maxiters = 5000
                DU = DdaggerD(D(U))
                # warmup
                solve_dirac!(ψ, DU, ϕ, temp1, temp2, temp3; tol, maxiters)
                MetaQCD.Fields.clear!(ψ)
                mpi_barrier()
                println("Benchmarking $(opname) $(tstring) ($(mpi_myrank()))")
                stats = @timed solve_dirac!(ψ, DU, ϕ, temp1, temp2, temp3; tol, maxiters)
                iters, _ = stats.value
                @show iters
                flops = prod(global_dims) * FLOPS[opname] * iters / 1e9
                mem = prod(global_dims) * mem_per_site(opname, T, 1) * iters / 1e9
                _time = stats.time
                perf = flops / _time
                bw = mem / _time

                if mpi_amroot()
                    benchprint(fp, "$(opname) $(tstring)")
                    str = @sprintf(
                        "%-15s%-12.3f%-12s%-12s%-10s[GFLOPs]",
                        "", perf, "-", "-", "-"
                    )
                    benchprint(fp, str)
                    str = @sprintf(
                        "%-15s%-12.3f%-12s%-12s%-10s[GB/s]",
                        "", bw, "-", "-", "-"
                    )
                    benchprint(fp, str)
                end

                mpi_barrier()
                continue
            else
                staggered = opname == "Staggered"
                csw = opname == "Wilson-Clover" ? 1.78 : 0.0
                ϕ = Spinorfield(U; staggered)
                ψ = Spinorfield(U; staggered)
                D = operator(U, 0.01; csw=csw)
                # warmup
                mul!(ψ, (D(U)), ϕ)
                mpi_barrier()
                println("Benchmarking $(opname) $(tstring) Operator ($(mpi_myrank()))")
                bench = @be _ $bench_mul!($B, $ψ, $(D)($U), $ϕ, $nlaunches) mpi_barrier() evals=2 samples=100 seconds=10
            end

            flops = prod(global_dims) * FLOPS[opname] / 1e9
            mem = prod(global_dims) * mem_per_site(opname, T, N) / 1e9

            mintime = minimum(s.time for s in bench.samples) / nlaunches
            mediantime = median(s.time for s in bench.samples) / nlaunches
            meantime = mean(s.time for s in bench.samples) / nlaunches
            stdtime = std(s.time for s in bench.samples) / nlaunches

            maxperf = flops / mintime
            medianperf = flops / mediantime
            meanperf = flops / meantime
            stdperf = meanperf - (flops / (meantime+stdtime))

            maxbw = mem / mintime
            medianbw = mem / mediantime
            meanbw = mem / meantime
            stdbw = meanbw - (mem / (meantime+stdtime))

            if mpi_amroot()
                benchprint(fp, "$(opname) $(tstring)")
                str = @sprintf(
                    "%-15s%-12.3f%-12.3f%-12.3f%-10.3f[GFLOPs]",
                    "", maxperf, medianperf, meanperf, stdperf
                )
                benchprint(fp, str)
                str = @sprintf(
                    "%-15s%-12.3f%-12.3f%-12.3f%-10.3f[GB/s]",
                    "", maxbw, medianbw, meanbw, stdbw
                )
                benchprint(fp, str)
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

function mem_per_site(op, ::Type{T}, nfloat) where T
    # 1 read fermion on site
    # 8 reads fermion neighbors in all μ-directions forward and backward
    # 1 write fermion on site
    # 8 reads gauge neighbors in all μ-directions forward and backward
    if op == "Staggered"
        return sizeof(Complex{T}) * (10 * 3 + 8 * nfloat/2)
    elseif op == "Wilson"
        return sizeof(Complex{T}) * (10 * 12 + 8 * nfloat/2)
    elseif op == "Wilson-Clover"
        # 1 read fermion on site
        # 1 write fermion on site
        # 6 x 16 read gauge for clover
        return sizeof(Complex{T}) * (2 * 12 + 6 * (16 * nfloat/2))
    elseif op == "Copy"
        return sizeof(Complex{T}) * (2 * 4 * nfloat/2)
        # return sizeof(Complex{T}) * (2 * 12)
    elseif op == "Halo-Exchange"
        return sizeof(Complex{T}) * (2 * 4 * nfloat/2)
        # return sizeof(Complex{T}) * (2 * 12)
    elseif op == "Dot-Staggered"
        return sizeof(Complex{T}) * (2 * 3)
    elseif op == "Dot-Wilson"
        return sizeof(Complex{T}) * (2 * 12)
    elseif op == "Invert-Staggered"
        mem_op = sizeof(Complex{T}) * (10 * 3 + 8 * nfloat/2)
        mem_axpy = 2sizeof(Complex{T}) * (2 * 3)
        mem_axpby = sizeof(Complex{T}) * (2 * 3)
        mem_dot = 2sizeof(Complex{T}) * (2 * 3)
        return mem_op + mem_axpy + mem_axpby + mem_dot
    elseif op == "Invert-Wilson"
        mem_op = sizeof(Complex{T}) * (2 * 12 + 6 * (16 * nfloat/2))
        mem_axpy = 2sizeof(Complex{T}) * (2 * 12)
        mem_axpby = sizeof(Complex{T}) * (2 * 12)
        mem_dot = 2sizeof(Complex{T}) * (2 * 12)
        return mem_op + mem_axpy + mem_axpby + mem_dot
    else
        error()
    end
end

function benchprint(io, str)
    println(io, str)
    println(str)
    return nothing
end

main()


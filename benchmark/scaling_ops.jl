using MetaQCD, MetaQCD.Utils, MPI, LinearAlgebra, Chairmarks
using Random, Statistics, Printf
using AMDGPU
using MetaQCD.DiracOperators: solve_dirac!
using MetaQCD.Fields: update_halo!, validate_halo!
using DelimitedFiles

const FLOPS = [
    # "Copy-Gauge" 0
    # "Copy-Staggered" 0
    # "Copy-Wilson" 0
    # "Dot-Staggered" 22
    # "Dot-Wilson" 94
    "Staggered" 587
    # "Staggered-EO" 587/2
    # "Wilson" 1368
    # "Wilson-Clover" 1368 + 1728
    # "Invert-Staggered" 587 + 2*22 + 2*12 + 18 # op + 2dot + 2axpy + axpby
    # "Invert-Wilson" 1368 + 2*22 + 2*12 + 18 # op + 2dot + 2axpy + axpby
    # "Halo-Exchange" 0
]

function bench_mul!(::Type{B}, ϕ, D, ψ, nlaunches; do_barrier=false) where B
    for _ in 1:nlaunches
        mul!(ψ, D, ϕ)
    end
    MetaQCD.Fields.device_synchronize(B())
    do_barrier && mpi_barrier()
    return nothing
end

function bench_copy!(::Type{B}, a, b, nlaunches; do_barrier=false) where B
    for _ in 1:nlaunches
        copy!(a, b)
    end
    MetaQCD.Fields.synchronize(B())
    do_barrier && mpi_barrier()
    return nothing
end

function bench_dot(::Type{B}, a, b, nlaunches; do_barrier=false) where B
    for _ in 1:nlaunches
        dot(a, b)
    end
    MetaQCD.Fields.synchronize(B())
    do_barrier && mpi_barrier()
    return nothing
end

function bench_exchange(::Type{B}, u, nlaunches; do_barrier=false) where B
    for _ in 1:nlaunches
        update_halo!((u,))
    end
    MetaQCD.Fields.synchronize(B())
    do_barrier && mpi_barrier()
    return nothing
end

function main(; strong=false)
    MetaQCD.MetaIO.set_global_logger!(1, nothing)
    nlaunches = 10
    B = ROCBackend
    GA = WilsonGaugeAction
    N = 12
    halo_width = 1

    if strong
        # global_dims = (96, 96, 96, 96)
        global_dims = (70, 70, 70, 20)
        numprocs_cart = distribute_procs(global_dims, mpi_size())
        # numprocs_cart = (1, 2, 2, 4)
    else
        numprocs_cart = distribute_procs_capped(16, mpi_size())
        global_dims = (64, 64, 64, 64) .* numprocs_cart
    end

    extra_dir = "22-06-26_overlap"
    result_dir = joinpath(@__DIR__, "scaling_results", extra_dir)
    time_dir = joinpath(@__DIR__, "scaling_results", extra_dir, "timings")

    if mpi_amroot()
        @show global_dims, numprocs_cart
    end
    mpi_barrier()
    bstring = lowercase(string(B))
    pstring = string(numprocs_cart...)
    hstring = MetaQCD.Fields.HIDE_COMMS == Val(true) ? "_hide" : ""
    sstring = strong ? "_strong" : "_weak"
    lstring = "$(global_dims[1])x$(global_dims[2])x$(global_dims[3])x$(global_dims[4])"
    npstring = "$(numprocs_cart[1])-$(numprocs_cart[2])-$(numprocs_cart[3])-$(numprocs_cart[4])"
    filename = "$(bstring)_$(lstring)_$(npstring)procs_$(pstring)$(hstring)$(sstring).txt" 
    timingname = joinpath(time_dir, "$(bstring)_$(lstring)_$(npstring)procs_$(pstring)$(hstring)$(sstring)")
    ispath(time_dir) || mkpath(time_dir)

    if mpi_amroot()
        fp = open(joinpath(result_dir, filename), "w")
        println(
            fp,
            rpad("", 15),
            rpad("max", 12),
            rpad("median", 12),
            rpad("mean", 12),
            rpad("std", 10),
        )
    end
    mpi_barrier()

    deviceid_printed = false

    for iop in axes(FLOPS, 1)
        opname = FLOPS[iop, 1]
        timings_barrier = []
        timings_nobarrier = []
        for T in (Float16, Float32, Float64)
            tstring = lowercase(string(T))
            U = Gaugefield{B,T,GA,N}(global_dims..., 6.0; numprocs_cart, halo_width)
            validate_halo!(U)

            if !deviceid_printed
                for irank in 0:mpi_size()-1
                    if mpi_myrank() == irank
                        println("$(mpi_myrank()): device $(AMDGPU.device_id())")
                        mpi_barrier()
                    end
                end
                deviceid_printed = true
            end

            if opname == "Copy-Gauge"
                a = similar(U)
                b = similar(U)
                # warmup
                copy!(a, b)
                MetaQCD.Fields.synchronize(B())
                mpi_barrier()
                println("Benchmarking $(opname) $(tstring) ($(mpi_myrank()))")
                bench_barrier = @be _ $bench_copy!($B, $a, $b, $nlaunches; do_barrier=true) evals=1 samples=100 seconds=10
                bench_nobarrier = @be _ $bench_copy!($B, $a, $b, $nlaunches; do_barrier=false) evals=1 samples=100 seconds=10
            elseif opname == "Copy-Staggered"
                a = Spinorfield(U; staggered=true)
                b = Spinorfield(U; staggered=true)
                # warmup
                copy!(a, b)
                MetaQCD.Fields.synchronize(B())
                mpi_barrier()
                println("Benchmarking $(opname) $(tstring) ($(mpi_myrank()))")
                bench_barrier = @be _ $bench_copy!($B, $a, $b, $nlaunches; do_barrier=true) evals=1 samples=100 seconds=10
                bench_nobarrier = @be _ $bench_copy!($B, $a, $b, $nlaunches; do_barrier=false) evals=1 samples=100 seconds=10
            elseif opname == "Copy-Wilson"
                a = Spinorfield(U)
                b = Spinorfield(U)
                # warmup
                copy!(a, b)
                MetaQCD.Fields.synchronize(B())
                mpi_barrier()
                println("Benchmarking $(opname) $(tstring) ($(mpi_myrank()))")
                bench_barrier = @be _ $bench_copy!($B, $a, $b, $nlaunches; do_barrier=true) evals=1 samples=100 seconds=10
                bench_nobarrier = @be _ $bench_copy!($B, $a, $b, $nlaunches; do_barrier=false) evals=1 samples=100 seconds=10
            elseif opname == "Halo-Exchange"
                f = Spinorfield(U; staggered=true)
                # warmup
                update_halo!((f,))
                MetaQCD.Fields.synchronize(B())
                mpi_barrier()
                println("Benchmarking $(opname) $(tstring) ($(mpi_myrank()))")
                bench_barrier = @be _ $bench_exchange($B, $f, $nlaunches; do_barrier=true) evals=1 samples=100 seconds=10
                bench_nobarrier = @be _ $bench_exchange($B, $f, $nlaunches; do_barrier=false) evals=1 samples=100 seconds=10
            elseif contains(opname, "Dot")
                staggered = contains(opname, "Staggered")
                a = Spinorfield(U; staggered)
                b = Spinorfield(U; staggered)
                # warmup
                dot(a, b)
                MetaQCD.Fields.synchronize(B())
                mpi_barrier()
                println("Benchmarking $(opname) $(tstring) ($(mpi_myrank()))")
                bench_barrier = @be _ $bench_dot($B, $a, $b, $nlaunches; do_barrier=true) evals=1 samples=100 seconds=10
                bench_nobarrier = @be _ $bench_dot($B, $a, $b, $nlaunches; do_barrier=false) evals=1 samples=100 seconds=10
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
                MetaQCD.Fields.synchronize(B())
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
                if opname == "Staggered"
                    D = StaggeredDiracOperator(U, 0.01)
                    ϕ = Spinorfield(U; staggered=true)
                    ψ = Spinorfield(U; staggered=true)
                elseif opname == "Staggered-EO"
                    D = StaggeredEOPreDiracOperator(U, 0.0)
                    NX, NY, NZ, NT = global_dims
                    ϕ = Spinorfield(U; staggered=true)
                    ψ = Spinorfield(U; staggered=true)
                    # ϕ = Spinorfield{B,T,1}(NX, NY, NZ, NT÷2)
                    # ψ = Spinorfield{B,T,1}(NX, NY, NZ, NT÷2)
                else
                    csw = opname == "Wilson-Clover" ? 1.78 : 0.0
                    D = WilsonDiracOperator(U, 0.01; csw=csw)
                    ϕ = Spinorfield(U)
                    ψ = Spinorfield(U)
                end
                # warmup
                bench_mul!(B, ψ, (D(U)), ϕ, 1)
                mpi_barrier()
                println("Benchmarking $(opname) $(tstring) Operator ($(mpi_myrank()))")
                bench_barrier = @be _ $bench_mul!($B, $ψ, $(D)($U), $ϕ, $nlaunches; do_barrier=true) evals=1 samples=50 seconds=100
                bench_nobarrier = @be _ $bench_mul!($B, $ψ, $(D)($U), $ϕ, $nlaunches; do_barrier=false) evals=1 samples=50 seconds=100
            end

            flops = prod(global_dims) * FLOPS[iop, 2] / 1e9
            mem = prod(global_dims) * mem_per_site(opname, T, N) / 1e9
            if mpi_amroot()
                push!(timings_barrier, [s.time for s in bench_barrier.samples])
                push!(timings_nobarrier, [s.time for s in bench_nobarrier.samples])
                if T == Float64 
                    open(timingname*"_$(opname)_barrier", "w") do io
                        writedlm(io, zip(timings_barrier...), '\t')
                    end
                    open(timingname*"_$(opname)_nobarrier", "w") do io
                        writedlm(io, zip(timings_nobarrier...), '\t')
                    end
                end
            end
            mpi_barrier()

            if mpi_amroot()
                for (mode, bench) in (("barrier", bench_barrier), ("nobarrier", bench_nobarrier))
                    mintime = minimum(s.time for s in bench.samples) / nlaunches
                    mediantime = median(s.time for s in bench.samples) / nlaunches
                    meantime = mean(s.time for s in bench.samples) / nlaunches
                    stdtime = std(s.time for s in bench.samples) / nlaunches

                    maxperf = flops / mintime
                    medianperf = flops / mediantime
                    meanperf = flops / meantime
                    stdperf = meanperf - (flops / (meantime + stdtime))

                    maxbw = mem / mintime
                    medianbw = mem / mediantime
                    meanbw = mem / meantime
                    stdbw = meanbw - (mem / (meantime + stdtime))

                    benchprint(fp, "$(opname) $(tstring) [$(mode)]")
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
                flush(fp)
            end
        end
    end

    mpi_amroot() && close(fp)
    mpi_barrier()
    return nothing
end

function distribute_procs(global_dims, numprocs)
    start_proc = global_dims[1] >= global_dims[4] ? 1 : 4
    end_proc = global_dims[1] >= global_dims[4] ? 4 : 1
    sgn = start_proc == 1 ? 1 : -1
    procsleft = numprocs
    numprocs_cart = [1, 1, 1, 1]
    dim = start_proc
    while procsleft > 1
        old_procs = numprocs_cart[dim]
        new_procs = old_procs * 2

        if global_dims[dim]/old_procs % 2 != 0
            if dim == end_proc
                error("too many procs")
            else
                dim += sgn*1
            end
            continue
        end

        numprocs_cart[dim] = new_procs
        procsleft /= 2
        @assert global_dims[dim]/new_procs > 4 "too many procs"

        if global_dims[dim]/new_procs <= 8 || (global_dims[dim]/new_procs) % 2 != 0
            if dim == end_proc
                error("too many procs")
            else
                dim += sgn*1
            end
        end
    end
    return (numprocs_cart...,)
end

function distribute_procs_capped(cap, numprocs)
    procsleft = numprocs
    numprocs_cart = [1, 1, 1, 1]
    dim = 4
    while procsleft > 1
        numprocs_cart[dim] *= 2
        @assert numprocs_cart[dim] <= cap
        procsleft = procsleft ÷ 2

        if numprocs_cart[dim] == cap
            dim = dim == 1 ? 4 : dim - 1
        end
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
    elseif op == "Staggered-EO"
        return sizeof(Complex{T}) * (10 * 3 + 8 * nfloat/2) / 2
    elseif op == "Wilson"
        return sizeof(Complex{T}) * (10 * 12 + 8 * nfloat/2)
    elseif op == "Wilson-Clover"
        # 1 read fermion on site
        # 1 write fermion on site
        # 6 x 16 read gauge for clover
        clover_part = 6 * (16 * nfloat/2)
        wilson_part = 10 * 12 + 8 * nfloat/2 + 6 * 9 # last is the reads of the 6 fieldstrengthtensor entries per site
        return sizeof(Complex{T}) * (wilson_part + clover_part)
    elseif op == "Copy-Gauge"
        return sizeof(Complex{T}) * (2 * 4 * nfloat/2)
    elseif op == "Copy-Staggered"
        return sizeof(Complex{T}) * (2 * 3)
    elseif op == "Copy-Wilson"
        return sizeof(Complex{T}) * (2 * 12)
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

main(strong=true)

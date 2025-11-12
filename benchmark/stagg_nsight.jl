using AMDGPU
using MetaQCD, MetaQCD.Utils, MPI, LinearAlgebra, BenchmarkTools
using Random
using MetaQCD.Fields: update_halo!, parallelfor, start_halo_update!, finalize_halo_update!
using MetaQCD.Fields: shrink_bulk, device_to_host, _parallelfor, synchronize, get_stream, get_priority_stream, default_stream
using MetaQCD.DiracOperators: staggered_kernel

function main()
    # numprocs_cart = (1, 1, 1, mpi_size())
    numprocs_cart = (1, 1, 2, 2)
    halo_width = 1
    B = ROCBackend
    T = Float32
    U = Gaugefield{B,T,WilsonGaugeAction,12}(64, 64, 64, 64, 6.0; numprocs_cart, halo_width)
    ϕ = Spinorfield(U; staggered=true); ψ = Spinorfield(U; staggered=true);
    D = StaggeredDiracOperator(U, 0.01);
    mass = D.mass
    bc = D.boundary_condition

    # mul!(ψ, D(U), ϕ)
    # mul!(ψ, D(U), ϕ)
    #
    # random_gauges!(U)
    # gaussian_pseudofermions!(ϕ)

    for i in 1:10
        tinner_start = time_ns()
        inner_bulk = shrink_bulk(eachindex(U, ϕ, ψ), halo_width)

        _parallelfor((U, ϕ, ψ), inner_bulk, B, 256; stream=get_stream(B(), 1)) do site, (U, ϕ, ψ)
            ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, Float32, false)
        end

        start_halo_update!((U, ϕ); do_edges=Val(true))

        touter_start = time_ns()

        outer_bulk = U.topology.border_iterators

        _parallelfor((U, ϕ, ψ), outer_bulk, B, 256) do site, (U, ϕ, ψ)
            ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, Float32, false)
        end
        touter_stop = time_ns()

        synchronize(B(), get_stream(B(), 1))
        tinner_stop = time_ns()

        tinner = (tinner_stop - tinner_start) / 1e9
        touter = (touter_stop - touter_start) / 1e9
        if mpi_amroot()
            println("==== #$i ====")
            @show tinner
            @show touter
            @show tinner + touter
            println("=============")
        end
    end

    for i in 1:10
        tinner_start = time_ns()
        inner_bulk = shrink_bulk(eachindex(U, ϕ, ψ), halo_width)

        _parallelfor((U, ϕ, ψ), inner_bulk, B, 256) do site, (U, ϕ, ψ)
            ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, Float32, false)
        end

        start_halo_update!((U, ϕ); do_edges=Val(false))

        touter_start = time_ns()

        outer_bulk = U.topology.border_iterators

        _parallelfor((U, ϕ, ψ), outer_bulk, B, 256) do site, (U, ϕ, ψ)
            ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, Float32, false)
        end
        touter_stop = time_ns()

        synchronize(B(), default_stream(B()))
        tinner_stop = time_ns()

        tinner = (tinner_stop - tinner_start) / 1e9
        touter = (touter_stop - touter_start) / 1e9
        if mpi_amroot()
            println("==== #$i ====")
            @show tinner
            @show touter
            @show tinner + touter
            println("=============")
        end
    end

    MPI.Finalize()
end

main()
ENABLE_JITPROFILING=1 mpirun -ppn 4 -np 8 --cpu-bind list:2-23:26-47:50-71:74-95 rocprofv2 --plugin perfetto --hip-trace --hsa-trace --kernel-trace -o prof -d rocprof_reports/ julia +1.12.0-rc3 -t auto --project --startup-file=no benchmark/stagg_nsight.jl

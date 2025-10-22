using AMDGPU
using MetaQCD, MetaQCD.Utils, MPI, LinearAlgebra, BenchmarkTools
using Random
using NVTX
using Colors
using MetaQCD.Fields: update_halo!, parallelfor, start_halo_update!, finalize_halo_update!
using MetaQCD.Fields: shrink_bulk, device_to_host, _parallelfor
using MetaQCD.DiracOperators: staggered_kernel

function main()
    # ------- NVTX -------
    NVTX.enable_gc_hooks()
    # ------- NVTX -------

    numprocs_cart = (1, 1, 1, 8)
    halo_width = 1
    B = ROCBackend
    T = Float32
    U = Gaugefield{B,T,WilsonGaugeAction,12}(32, 32, 32, 64, 6.0; numprocs_cart, halo_width)
    ϕ = Spinorfield(U; staggered=true); ψ = Spinorfield(U; staggered=true);
    D = StaggeredDiracOperator(U, 0.01);
    mul!(ψ, D(U), ϕ)

    mass = D.mass
    bc = D.boundary_condition

    random_gauges!(U)
    gaussian_pseudofermions!(ϕ)

    # ------- NVTX -------
    nvtx_range = NVTX.range_start(; message="staggered mul!", color=colorant"yellow")
    # ------- NVTX -------
    
    NVTX.range_push(; message="inner update + exchange", color=colorant"blue")
    reqs = start_halo_update!((U, ϕ); do_edges=Val(false))
    t = @task finalize_halo_update!(reqs, (U, ϕ), B)
    schedule(t)
    inner_bulk = shrink_bulk(eachindex(U, ϕ, ψ), halo_width)
    # bulk = eachindex(U, ϕ, ψ)

    _parallelfor((U, ϕ, ψ), inner_bulk, B, 256) do site, (U, ϕ, ψ)
        ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, Float32, false)
    end

    wait(t)
    # finalize_halo_update!(reqs, sr, (U, ϕ), B)
    NVTX.range_pop()

    NVTX.range_push(; message="outer update", color=colorant"red")
    outer_bulk = U.topology.border_iterators

    # _parallelfor((U, ϕ, ψ), bulk, B, 256) do site, (U, ϕ, ψ)
    #     ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, Float32, false)
    # end
    _parallelfor((U, ϕ, ψ), outer_bulk, B, 256) do site, (U, ϕ, ψ)
        ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, Float32, false)
    end
    NVTX.range_pop()

    # ------- NVTX -------
    NVTX.range_end(nvtx_range)
    # ------- NVTX -------

    MPI.Finalize()
end

main()

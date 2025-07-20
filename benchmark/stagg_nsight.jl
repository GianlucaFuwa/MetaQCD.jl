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

    numprocs_cart = (1, 2, 2, 1)
    halo_width = 1
    B = CPU
    T = Float64
    U = Gaugefield{B,T,WilsonGaugeAction}(32, 32, 32, 32, 6.0; numprocs_cart, halo_width)
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
    sendrecvtasks = start_halo_update!((U, ϕ); do_edges=Val(true))
    inner_bulk = shrink_bulk(eachindex(U, ϕ, ψ), halo_width)

    _parallelfor((U, ϕ, ψ), inner_bulk, B, 256) do site, (U, ϕ, ψ)
        ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, T, false)
    end

    finalize_halo_update!(sendrecvtasks)
    NVTX.range_pop()

    NVTX.range_push(; message="outer update", color=colorant"red")
    outer_bulk = U.topology.flat_border_sites

    _parallelfor((U, ϕ, ψ), outer_bulk, B, 256) do site, (U, ϕ, ψ)
        ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, T, false)
    end
    NVTX.range_pop()

    # ------- NVTX -------
    NVTX.range_end(nvtx_range)
    # ------- NVTX -------

    # sout = mpi_amroot() ? stdout : devnull
    # redirect_stdout(sout) do
    #     _b = @benchmark mul!($ψ, $(D(U)), $ϕ)
    #     io = IOBuffer()
    #     show(io, "text/plain", _b)
    #     s = String(take!(io))
    #     println(s)
    # end
    MPI.Finalize()
end

main()

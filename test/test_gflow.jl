function test_gradflow(backend=CPU; nprocs_cart=(1, 1, 1, 1), halo_width=1)
    Random.seed!(123)
    println("Smearing tests")
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    U = Gaugefield{CPU,Float64,WilsonGaugeAction}(NX, NY, NZ, NT, 6.0, nprocs_cart, halo_width)

    filename = if MetaQCD.Fields.is_distributed(U)
        pkgdir(MetaQCD, "test", "testconf_mpi")
    else
        pkgdir(MetaQCD, "test", "testconf.txt")
    end

    load_config!(BridgeFormat(), U, filename)

    if backend !== CPU
        U = MetaQCD.to_backend(backend, U)
    end

    mfac = 1 / (6 * U.NV * U.NC)
    plaq = plaquette_trace_sum(U) * mfac

    g = GradientFlow(U, "euler", 3, 1, 0.12)
    s = StoutSmearing(U, 3, 0.12)

    copy!(g.Uflow, U)

    mpi_amroot() && println("0\tplaq: $plaq")

    p_flow = zeros(3)

    for iflow in 1:g.numflow
        flow!(g)
        plaq = plaquette_trace_sum(g.Uflow) * mfac
        mpi_amroot() && println("$iflow\tplaq (gflow): $plaq")
        p_flow[iflow] = plaq
    end

    calc_smearedU!(s, U)
    p_stout = plaquette_trace_sum(s.Usmeared_multi[end]) * mfac
    mpi_amroot() && println("3\tplaq (stout): $(p_stout)\n")

    mpi_amroot() && (@test isapprox(p_stout, p_flow[end]))
    return isapprox(p_stout, p_flow[end])
end

function test_gradflow(; backend=CPU, nprocs_cart=(1, 1, 1, 1), halo_width=1)
    Random.seed!(123)
    println("Smearing tests")
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    U = Gaugefield{CPU,Float64,WilsonGaugeAction}(
        NX, NY, NZ, NT, 6.0, numprocs_cart=nprocs_cart, halo_width=halo_width
    )
    numflow = 7

    filename = if nprocs_cart != (1, 1, 1, 1)
        pkgdir(MetaQCD, "test", "testconf_mpi")
    else
        pkgdir(MetaQCD, "test", "testconf.txt")
    end

    load_field!(BridgeFormat(), U, filename)

    if backend !== CPU
        U = MetaQCD.to_backend(backend, U)
    end

    mfac = 1 / (18 * length(U))
    plaq = plaquette_trace_sum(U) * mfac

    g = GradientFlow(U; integrator="euler", numflow=numflow, steps=1, tf=0.12)
    s = StoutSmearing(U; numlayers=numflow, rho=0.12)

    copy!(g.Uflow, U)

    mpi_amroot() && println("0\tplaq: $plaq\n")

    p_flow = zeros(numflow)

    for iflow in 1:g.numflow
        flow!(g)
        plaq = plaquette_trace_sum(g.Uflow) * mfac
        mpi_amroot() && println("$iflow\tplaq (gflow): $plaq")
        p_flow[iflow] = plaq
    end

    println()
    calc_smearedU!(s, U)
    p_stout = plaquette_trace_sum(s.Usmeared_multi[end]) * mfac

    for i in eachindex(s.Usmeared_multi)
        i == 1 && continue
        p = plaquette_trace_sum(s.Usmeared_multi[i]) * mfac
        mpi_amroot() && println("$(i-1)\tplaq (stout): $(p)")
    end

    if mpi_amroot()
        @testset "Gradient flow / Stout equivalence" begin
            @test isapprox(p_stout, p_flow[end])
        end
    end
    return isapprox(p_stout, p_flow[end])
end

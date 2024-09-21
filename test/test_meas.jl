const PLAQ_EXP = 0.587818337847024
const POLY_EXP = 0.5255246068616176 - 0.15140850971249734im
const TOPO_EXP = Dict(
    "plaquette" => -0.2730960126400261,
    "clover" => -0.027164585971545994,
    "improved" => -0.03210085960569041,
)

function test_measurements(backend=CPU; nprocs_cart=(1, 1, 1, 1), halo_width=1)
    println("Gauge observable tests")
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

    m_plaq = PlaquetteMeasurement(U)
    plaq = measure(m_plaq, U, 1, 1)

    mpi_amroot() && println("==========")

    m_poly = PolyakovMeasurement(U)
    poly = measure(m_poly, U, 1, 1)

    mpi_amroot() && println("==========")

    m_wilson = WilsonLoopMeasurement(U)
    wilsonloop = measure(m_wilson, U, 1, 1)

    mpi_amroot() && println("==========")

    TC_methods  = ["plaquette", "clover", "improved"]
    m_topo = TopologicalChargeMeasurement(U, TC_methods=TC_methods)
    topo = measure(m_topo, U, 1, 1)

    mpi_amroot() && println("==========")

    ED_methods  = ["plaquette", "clover", "improved"]
    m_ed = EnergyDensityMeasurement(U, ED_methods=ED_methods)
    ed = measure(m_ed, U, 1, 1)

    mpi_amroot() && println("==========")

    GA_methods = ["wilson", "symanzik_tree", "iwasaki", "dbw2"]
    m_gaction = GaugeActionMeasurement(U, GA_methods=GA_methods)
    gaction = measure(m_gaction, U, 1, 1)

    if mpi_amroot()
        @testset "Gauge observables" begin
            @test isapprox(PLAQ_EXP, plaq)
            nprocs_cart[4] == 1 && (@test isapprox(POLY_EXP, poly)) # FIXME: for now U cannot be partitioned in time dimension
            @test isapprox(TOPO_EXP["plaquette"], topo["plaquette"])
            @test isapprox(TOPO_EXP["clover"], topo["clover"])
            @test isapprox(TOPO_EXP["improved"], topo["improved"])
            @test isapprox(PLAQ_EXP, wilsonloop[1, 1])
        end
    end

    mpi_barrier()
    return nothing
end

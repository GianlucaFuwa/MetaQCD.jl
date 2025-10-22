using MetaQCD
using MetaQCD.Utils
using Test
using LinearAlgebra
using Random
using AMDGPU

const EXP4 = Dict(
    "plaq" => 0.587818337847024,
    "poly" => 0.5255246068616176 - 0.15140850971249734im,
    "topo_plaq" => -0.2730960126400261,
    "topo_clov" => -0.027164585971545994,
    "topo_imp" => -0.03210085960569041,
)

const EXP16 = Dict(
    "plaq" => 0.5943106319764989,
    "poly" => 0.004036670632078757 + 0.009469086655463761im,
    "topo_plaq" => -14.13722570213623,
    "topo_clov" => 1.7354903168411089,
    "topo_imp" => 2.613871310127444,
)

function test_measurements(; backend=CPU, nprocs_cart=(1, 1, 1, 1), halo_width=2)
    mpi_amroot() && println("Gauge observable tests")
    if mpi_size() > 1
        NX = NY = NZ = NT = 16
    else
        NX = NY = NZ = NT = 16
    end
    U = Gaugefield{backend,Float64,WilsonGaugeAction,12}(
        NX, NY, NZ, NT, 6.0; numprocs_cart=nprocs_cart, halo_width=halo_width
    )

    add_str = mpi_size() > 1 ? "_16" : "_16"
    filename = if nprocs_cart != (1, 1, 1, 1)
        pkgdir(MetaQCD, "test", "testconf$(add_str)_mpi")
    else
        pkgdir(MetaQCD, "test", "testconf$(add_str).txt")
    end

    load_field!(BridgeFormat(), U, filename)

    m_plaq = PlaquetteMeasurement(U)
    plaq = measure(m_plaq, U)

    mpi_amroot() && println("==========")

    if nprocs_cart[4] == 1
        m_poly = PolyakovMeasurement(U)
        poly =  measure(m_poly, U)

        mpi_amroot() && println("==========")
    end

    if nprocs_cart == (1, 1, 1, 1)
        m_wilson = WilsonLoopMeasurement(U)
        wilsonloop = measure(m_wilson, U)

        mpi_amroot() && println("==========")
    end

    TC_methods  = if mpi_size() > 1 && halo_width < 2
        ["plaquette", "clover"]
    else
        ["plaquette", "clover", "improved"]
    end
    m_topo = TopologicalChargeMeasurement(U, TC_methods=TC_methods)
    topo = measure(m_topo, U)

    mpi_amroot() && println("==========")

    ED_methods  = if mpi_size() > 1 && halo_width < 2
        ["plaquette", "clover"]
    else
        ["plaquette", "clover", "improved"]
    end
    m_ed = EnergyDensityMeasurement(U, ED_methods=ED_methods)
    ed = measure(m_ed, U)

    mpi_amroot() && println("==========")

    GA_methods = ["wilson", "symanzik_tree", "iwasaki", "dbw2"]
    m_gaction = GaugeActionMeasurement(U, GA_methods=GA_methods)
    gaction = measure(m_gaction, U)

    if mpi_amroot()
        expvalues = NX == 4 ? EXP4 : EXP16
        @testset "Gauge observables" begin
            @test isapprox(expvalues["plaq"], plaq)
            nprocs_cart[4] == 1 && (@test isapprox(expvalues["poly"], poly)) # FIXME: for now U cannot be partitioned in time dimension
            @test isapprox(expvalues["topo_plaq"], topo["plaquette"])
            @test isapprox(expvalues["topo_clov"], topo["clover"])
            halo_width >= 2 && (@test isapprox(expvalues["topo_imp"], topo["improved"]))
            if nprocs_cart == (1, 1, 1, 1)
                @test isapprox(expvalues["plaq"], wilsonloop[1, 1])
            end
        end
    end

    mpi_barrier()
    return nothing
end

test_measurements(; backend=ROCBackend, nprocs_cart=(1, 1, 1, mpi_size()))
# test_measurements(nprocs_cart=(1, 2, 1, 2))
# test_measurements(nprocs_cart=(2, 1, 1, 2))
# test_measurements(nprocs_cart=(1, 2, 2, 1))
# test_measurements(nprocs_cart=(2, 2, 1, 1))
# test_measurements(nprocs_cart=(2, 1, 2, 1))

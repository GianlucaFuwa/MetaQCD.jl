using MetaQCD
using MetaQCD.Utils
using MetaQCD.Updates: set_instanton!
using MetaQCD.Measurements: top_charge
using Random

function test_instanton(backend=CPU; nprocs_cart=(1, 1, 1, 1), halo_width=1)
    Random.seed!(123)
    println("Instanton tests")
    N = 16
    U = Gaugefield{backend,Float64,WilsonGaugeAction,12}(
        N, N, N, N, 6.0, numprocs_cart=nprocs_cart, halo_width=halo_width
    )
    g = GradientFlow(U; integrator="euler", numflow=30, steps=1, tf=0.12)

    if backend !== CPU
        U = MetaQCD.convert_field(backend, U)
    end

    for Q in -6:1:6
        mpi_amroot() && println("Q = $Q")

        set_instanton!(U, Q)
        copy!(g.Uflow, U)
        for _ in 1:g.numflow
            flow!(g)
        end
        mpi_amroot() && println("tclover: $(top_charge(g.Uflow, "clover"))\n")
    end

    return nothing
end

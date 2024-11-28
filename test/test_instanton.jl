using MetaQCD
using MetaQCD.Utils
using MetaQCD.Updates: set_instanton!
using MetaQCD.Measurements: top_charge
using Random

function test_instanton(backend=CPU; nprocs_cart=(1, 1, 1, 1), halo_width=1)
    Random.seed!(123)
    println("Instanton tests")
    NX = 12
    NY = 12
    NZ = 12
    NT = 12
    U = Gaugefield{CPU,Float64,WilsonGaugeAction}(NX, NY, NZ, NT, 6.0, nprocs_cart, halo_width)

    if backend !== CPU
        U = MetaQCD.to_backend(backend, U)
    end

    for Q in -6:1:6
        mpi_amroot() && println("Q = $Q")

        set_instanton!(U, Q)
        mpi_amroot() && println("tclover: $(top_charge(U, "improved"))\n")
    end

    return nothing
end

using MetaQCD
using MetaQCD.MPI
using Test

const COMM = MPI.COMM_WORLD
const MYRANK = MPI.Comm_rank(COMM)

function test_mpi()
    if MYRANK == 0
        println("test_mpi")
        @assert MPI.Comm_size(COMM) == 4
    end

    Ns = 4
    Nt = 4
    U = Gaugefield{CPU,Float64,WilsonGaugeAction}(Ns, Ns, Ns, Nt, 5.7, (1, 1, 2, 2), 1)
    filename = pkgdir(MetaQCD, "test", "testconf_mpi")
    load_config!(BridgeFormat(), U, filename, true)
    MetaQCD.Fields.update_halo!(U)

    factor = 1 / (6 * U.NV * U.NC)
    plaq = plaquette_trace_sum(U) * factor

    if MYRANK == 0
        isapprox(0.587818337847024, plaq)
        @show plaq
    end

    return nothing
end

test_mpi()

using MetaQCD
using MetaQCD.Utils
using LinearAlgebra
using Random
using Test

const PLAQ_EXP = 0.587818337847024
const POLY_EXP = 0.5255246068616176 - 0.15140850971249734im
const TOPO_EXP = Dict(
    "plaquette" => -0.2730960126400261,
    "clover" => -0.027164585971545994,
    "improved" => -0.03210085960569041,
)

function test_mpi()
    if mpi_amroot()
        println("test_mpi")
        # @assert mpi_size() == 4
    end

    Ns = 4
    Nt = 4
    U_22 = Gaugefield{CPU,Float64,WilsonGaugeAction}(Ns, Ns, Ns, Nt, 5.7, (2, 2, 1, 1), 1)
    U_4 = Gaugefield{CPU,Float64,WilsonGaugeAction}(Ns, Ns, Ns, Nt, 5.7, (4, 1, 1, 1), 1)
    U_w2 = Gaugefield{CPU,Float64,WilsonGaugeAction}(Ns, Ns, Ns, Nt, 5.7, (1, 1, 2, 2), 2)
    filename = pkgdir(MetaQCD, "test", "testconf_mpi")
    load_config!(BridgeFormat(), U_22, filename, true)
    load_config!(BridgeFormat(), U_4, filename, true)
    load_config!(BridgeFormat(), U_w2, filename, true)

    factor_22 = 1 / (6 * U_22.NV * U_22.NC)
    plaq_22 = plaquette_trace_sum(U_22) * factor_22
    factor_4 = 1 / (6 * U_4.NV * U_4.NC)
    plaq_4 = plaquette_trace_sum(U_4) * factor_4

    topop_22 = top_charge(U_22, "plaquette")
    topop_4 = top_charge(U_4, "plaquette")
    topoc_22 = top_charge(U_22, "clover")
    topoc_4 = top_charge(U_4, "clover")
    topoi_w2 = top_charge(U_w2, "improved")

    poly_22 = MetaQCD.Measurements.polyakov_traced(U_22)
    poly_4 = MetaQCD.Measurements.polyakov_traced(U_4)

    if mpi_amroot()
        @show isapprox(PLAQ_EXP, plaq_22)
        @show isapprox(PLAQ_EXP, plaq_4)
        @show isapprox(TOPO_EXP["plaquette"], topop_22)
        @show isapprox(TOPO_EXP["plaquette"], topop_4)
        @show isapprox(TOPO_EXP["clover"], topoc_22)
        @show isapprox(TOPO_EXP["clover"], topoc_4)
        @show isapprox(TOPO_EXP["improved"], topoi_w2)
        @show isapprox(POLY_EXP, poly_22)
        @show isapprox(POLY_EXP, poly_4)
    end

    return nothing
end

test_mpi()

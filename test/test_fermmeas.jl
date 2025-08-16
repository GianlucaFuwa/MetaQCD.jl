using MetaQCD

function test_fermion_measurements(backend=CPU)
    println("SU3testmeas")
    NX = NY = NZ = NT = 4
    U = Gaugefield{backend,Float64,WilsonGaugeAction,12}(NX, NY, NZ, NT, 6.0)
    filename = pkgdir(MetaQCD, "test", "testconf.txt")
    load_field!(BridgeFormat(), U, filename)

    m_pion = MetaQCD.Measurements.PionCorrelatorMeasurement(
        U; dirac_type="staggered", mass=1
    )
    pion = measure(m_pion, U)
    mpi_amroot() && println("PION CORR: $pion") # XXX: I think the reference value in QCDMeasurements.jl is for wilson operator
    return nothing
end

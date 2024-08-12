using MetaQCD

function test_fermion_measurements(backend=CPU)
    println("SU3testmeas")
    NX = NY = NZ = NT = 4
    U = Gaugefield{CPU,Float64,WilsonGaugeAction}(NX, NY, NZ, NT, 6.0)
    filename = pkgdir(MetaQCD, "test", "testconf.txt")
    # random_gauges!(U)
    load_config!(BridgeFormat(), U, filename)
    if backend !== CPU
        U = MetaQCD.to_backend(backend, U)
    end
    m_pion = MetaQCD.Measurements.PionCorrelatorMeasurement(
        U; dirac_type="wilson", mass=0.1
    )
    @time pion = get_value(measure(m_pion, U))
    println("PION CORR: $pion")
    return nothing
end

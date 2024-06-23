using MetaQCD

function SU3testfmeas(backend=CPU)
    println("SU3testmeas")
    Ns = 4
    Nt = 4
    U = Gaugefield{backend,Float64,WilsonGaugeAction}(NX, NY, NZ, NT, 6.0)
    filename = "./test/testconf.txt"
    # random_gauges!(U)
    loadU!(BridgeFormat(), U, filename)
    if backend !== nothing
        U = MetaQCD.to_backend(backend, U)
    end
    m_pion = MetaQCD.Measurements.PionCorrelatorMeasurement(
        U; dirac_type="wilson", mass=0.1
    )
    @time pion = get_value(measure(m_pion, U))
    println("PION CORR: $pion")
    return nothing
end

SU3testfmeas()

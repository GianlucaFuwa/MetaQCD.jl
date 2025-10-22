using MetaQCD
using MetaQCD.Utils
using Test
using LinearAlgebra
using Random
using AMDGPU

function test_gradflow(; backend=CPU, nprocs_cart=(1, 1, 1, 1), halo_width=1)
    Random.seed!(123)
    mpi_amroot() && println("Smearing tests")
    NX = NY = NZ = NT = 16
    U = Gaugefield{backend,Float64,WilsonGaugeAction,12}(
        NX, NY, NZ, NT, 6.0, numprocs_cart=nprocs_cart, halo_width=halo_width
    )
    numflow = 7

    filename = if nprocs_cart != (1, 1, 1, 1)
        pkgdir(MetaQCD, "test", "testconf_16_mpi")
    else
        pkgdir(MetaQCD, "test", "testconf_16.txt")
    end

    load_field!(BridgeFormat(), U, filename)

    mfac = 1 / (18 * length(U))
    plaq = plaquette_trace_sum(U) * mfac
    q = top_charge(U, "clover")

    g = GradientFlow(U; integrator="euler", numflow=numflow, steps=1, tf=0.12)
    s = StoutSmearing(U; numlayers=numflow, rho=0.12)

    copy!(g.Uflow, U)

    if mpi_amroot()
        println("0\tplaq: $plaq\n")
        println("0\tqclov: $q\n")
    end

    p_flow = zeros(numflow)
    q_flow = zeros(numflow)

    for iflow in 1:g.numflow
        flow!(g)
        plaq = plaquette_trace_sum(g.Uflow) * mfac
        q = top_charge(g.Uflow, "clover")
        mpi_amroot() && println("$(iflow)\tplaq (gflow): $(plaq)")
        mpi_amroot() && println("$(iflow)\tqclov (gflow): $(q)")
        p_flow[iflow] = plaq
        q_flow[iflow] = q
    end

    println()
    calc_smearedU!(s, U)
    p_stout = plaquette_trace_sum(s.Usmeared_multi[end]) * mfac
    q_stout = top_charge(s.Usmeared_multi[end], "clover")

    for i in eachindex(s.Usmeared_multi)
        # i == 1 && continue
        p = plaquette_trace_sum(s.Usmeared_multi[i]) * mfac
        q = top_charge(s.Usmeared_multi[i], "clover")
        mpi_amroot() && println("$(i-1)\tplaq (stout): $(p)")
        mpi_amroot() && println("$(i-1)\tqclov (stout): $(q)")
    end

    if mpi_amroot()
        @testset "Gradient flow / Stout equivalence" begin
            @test isapprox(p_stout, p_flow[end])
            @test isapprox(q_stout, q_flow[end])
        end
    end
    return isapprox(p_stout, p_flow[end])
end

test_gradflow(; backend=ROCBackend, nprocs_cart=(1, 1, 1, mpi_size()))

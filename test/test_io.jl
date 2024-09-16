using MetaQCD
using MetaQCD.MetaIO
using Test
using Random

function test_io()
    println("test_io")
    Random.seed!(1234)
    Ns = 4
    Nt = 4
    U = Gaugefield{CPU,Float64,WilsonGaugeAction}(Ns, Ns, Ns, Nt, 5.7)
    random_gauges!(U)
    tmpfile_bridge = tempname(pwd(); cleanup=true)
    tmpfile_jld = tempname(pwd(); cleanup=true)
    tmpfile_bmw = tempname(pwd(); cleanup=true)
    save_config(BridgeFormat(), U, tmpfile_bridge)
    save_config(JLD2Format(), U, tmpfile_jld)
    save_config(BMWFormat(), U, tmpfile_bmw)

    U_bridge = Gaugefield{CPU,Float64,WilsonGaugeAction}(Ns, Ns, Ns, Nt, 5.7)
    load_config!(BridgeFormat(), U_bridge, tmpfile_bridge)

    U_jld = Gaugefield{CPU,Float64,WilsonGaugeAction}(Ns, Ns, Ns, Nt, 5.7)
    load_config!(JLD2Format(), U_jld, tmpfile_jld)

    # U_bmw = Gaugefield{CPU,Float64,WilsonGaugeAction}(Ns, Ns, Ns, Nt, 5.7)
    # load_config!(BMWFormat(), U_bmw, tmpfile_bmw)

    site = CartesianIndex{4}(rand(1:Ns), rand(1:Ns), rand(1:Ns), rand(1:Nt))
    μ = rand(1:4)
    @test U[μ, site] ≈ U_bridge[μ, site]
    @test U[μ, site] ≈ U_jld[μ, site]
    # @test U[μ, site] ≈ U_bmw[μ, site]
    return nothing
end

using MetaQCD
using MetaQCD.MetaIO
using MetaQCD.Utils
using Test
using Random

function test_checkpoint()
    mpi_amroot() && println("Checkpoint test")
    checkpointer = Checkpointer(pwd(), 1)

    U = Gaugefield{CPU,Float64,WilsonGaugeAction}(4, 4, 4, 4, 6.0)
    Random.seed!(123)
    random_gauges!(U)
    univ = Univ(U, QuenchedFermionAction(), NoBias(), 1)
    updatemethod = HMC(univ.U, OMF4(), 1, 5)
    create_checkpoint(checkpointer, univ, updatemethod, nothing, 1)
    update!(updatemethod, univ.U)

    univ_args..., updatemethod_loaded, _, _ = load_checkpoint(pwd() * "/checkpoint_0")
    univ_loaded = Univ(univ_args...)
    update!(updatemethod_loaded, univ_loaded.U)

    rm(pwd() * "/checkpoint_0.jld2")
    
    if mpi_amroot()
        @testset "Checkpoint" begin
            @test calc_gauge_action(univ.U) ≈ calc_gauge_action(univ_loaded.U)
        end
    end

    return nothing
end

test_checkpoint()

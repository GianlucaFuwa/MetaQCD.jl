using Test
include("hunter_roofline.jl")
using .HunterRoofline

@testset "Hunter geometry and roofline" begin
    for filename in readdir(joinpath(@__DIR__, "scaling_results_hunter"))
        endswith(filename, ".txt") || continue
        geometry = benchmark_geometry(filename)
        @test prod(geometry.numprocs_cart) == geometry.nprocs
        @test all(geometry.global_dims .% geometry.numprocs_cart .== 0)
        if occursin("weak", filename)
            @test geometry.global_dims .÷ geometry.numprocs_cart == (64, 64, 64, 64)
        else
            @test geometry.global_dims == (96, 96, 96, 96)
        end
    end
    @test benchmark_geometry(
        "rocbackend_96x96x96x96_128procs_16811_strong.txt"
    ).numprocs_cart == (16, 8, 1, 1)
    @test_throws ErrorException benchmark_geometry("bad_name.txt")
    @test_throws ErrorException benchmark_geometry(
        "rocbackend_96x96x96x96_128procs_1111_strong.txt"
    )

    staggered = operator_work("Staggered float32")
    wilson = operator_work("Wilson float32")
    @test staggered.reported_bytes == 624
    @test wilson.reported_bytes == 1344
    @test staggered.intensity == 587 / 624
    @test wilson.spinor_bytes == 4staggered.spinor_bytes
    @test isnothing(operator_work("Wilson-Clover float32"))

    one = (; global_dims=(64, 64, 64, 64), numprocs_cart=(1, 1, 1, 1))
    ceiling = model_ceiling("Staggered float32", one)
    @test ceiling.effective_tb_s ≈ 5.3
    @test ceiling.tflops ≈ 5.3staggered.intensity
    @test ceiling.net_seconds == 0
    @test ceiling.sent_bytes_per_rank == 0

    # Eight contiguous ranks occupy two nodes. Only the two end faces per node
    # cross the NICs; four ranks do not each send two faces off-node.
    eight = (; global_dims=(64, 64, 64, 512), numprocs_cart=(1, 1, 1, 8))
    s = model_ceiling("Staggered float32", eight)
    @test s.local_dims == (64, 64, 64, 64)
    @test s.max_node_remote_bytes == 2 * 64^3 * 24
    @test s.net_seconds ≈ (2 * 64^3 * 24) / 100e9
    @test s.sent_bytes_per_rank == 2 * 64^3 * 24
    @test s.seconds ≈ s.memory_seconds + s.net_seconds
    @test s.boundary_fraction ≈ 2 / 64
    w = model_ceiling("Wilson float32", eight)
    @test w.net_seconds ≈ 4s.net_seconds
    @test w.sent_bytes_per_rank == 4s.sent_bytes_per_rank
    fast_net = model_ceiling(
        "Staggered float32",
        eight;
        hardware=HunterHardware(; network_bytes_s_per_node=200e9),
    )
    @test fast_net.net_seconds ≈ s.net_seconds / 2
    @test fast_net.tflops > s.tflops
    hidden = model_ceiling("Staggered float32", eight; overlap=true)
    @test hidden.seconds ≈ max(hidden.memory_seconds, hidden.net_seconds)
    @test hidden.tflops >= s.tflops
    cached = model_ceiling("Staggered float32", eight; refresh_halo=false)
    @test cached.net_seconds == 0
    @test cached.effective_tb_s ≈ 8 * 5.3
    @test cached.sent_bytes_per_rank == 0
    best_reuse = model_ceiling("Staggered float32", eight; hbm_traffic=:compulsory)
    @test best_reuse.tflops > s.tflops
    fp64 = model_ceiling("Staggered float64", eight)
    @test fp64.tflops ≈ s.tflops / 2
    all_remote = model_ceiling("Staggered float32", eight; placement=:all_remote)
    @test all_remote.net_seconds ≈ 4s.net_seconds
    @test_throws ErrorException model_ceiling(
        "Staggered float32", eight; hbm_traffic=:unknown
    )

    strong128 = (; global_dims=(96, 96, 96, 96), numprocs_cart=(16, 8, 1, 1))
    strong = model_ceiling("Staggered float32", strong128)
    @test strong.local_dims == (6, 12, 96, 96)
    @test strong.boundary_fraction ≈ 4 / 9
    # Extended faces include previous dimensions' halos, matching current buffers.
    face_only = model_ceiling("Staggered float32", strong128; extended_faces=false)
    @test face_only.sent_bytes_per_rank < strong.sent_bytes_per_rank
end

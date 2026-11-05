# Requires the plotting environment used for summarize_scaling_v2.jl.
using Test
include("summarize_scaling_v2.jl")

@testset "Scaling data and plot integration" begin
    dir = joinpath(@__DIR__, "scaling_results_hunter")
    strong, weak = collect_scaling_data(dir)
    @test length(strong) == length(weak) == 6
    for dataset in (strong, weak), data in values(dataset)
        @test data["nprocs"] == [2, 4, 8, 16, 32, 64, 128]
        @test length(data["geometry"]) == length(data["gauge_floats"]) == 7
    end
    @test weak["Staggered float32"]["bandwidth"][1] ≈ 6.550842
    @test weak["Staggered float32"]["bandwidth"][end] ≈ 88.861677
    @test weak["Wilson float32"]["bandwidth"][end] ≈ 102.626432
    _, median_weak = collect_scaling_data(dir; statistic=:median)
    @test median_weak["Staggered float32"]["bandwidth"][1] ≈ 5.876932
    @test_throws ErrorException collect_scaling_data(dir; statistic=:std)
    @test all(isempty, collect_build_scaling_data(dir))

    mktempdir() do temp
        # Test explicit-unit parsing with a zero FLOP/s Copy metric and metadata
        # that disambiguates a filename from older scaling_ops.jl versions.
        path = joinpath(temp, "rocbackend_8x8x8x8_1-1-1-2procs_1112_weak.txt")
        write(
            path,
            """
max median mean std
Copy-Staggered float32 [nobarrier]
  0 0 0 0 [GFLOPs]
  2000 1800 1700 20 [GB/s]
Staggered float32 [nobarrier]
  1000 900 800 10 [GFLOPs]
  2000 1800 1600 20 [GB/s]
""",
        )
        write(
            replace(path, ".txt" => ".toml"),
            """
global_dims = [8, 8, 8, 8]
numprocs_cart = [1, 1, 1, 2]
hide_communication = true
refresh_halo = true
gauge_floats = 18
""",
        )
        raw = parse_benchmark_file(path)
        @test only(raw["Copy-Staggered float32 [nobarrier]"]["gflops"]) == 0
        @test only(raw["Copy-Staggered float32 [nobarrier]"]["bandwidth"]) == 2
        _, data = collect_scaling_data(temp)
        @test data["Staggered float32"]["nprocs"] == [2]
        @test data["Staggered float32"]["overlap"] == [true]
        @test data["Staggered float32"]["gauge_floats"] == [18]
        out = joinpath(temp, "plots")
        plots = create_scaling_plots(Dict(), data, zeros(0, 2), zeros(0, 2); output_dir=out)
        @test length(plots) == 2
        @test isfile(joinpath(out, "weak_scaling_bandwidth.pdf"))
        @test isfile(joinpath(out, "weak_scaling_performance.pdf"))
        rows = readlines(joinpath(out, "scaling_model.csv"))
        @test length(rows) == 2
        @test split(rows[2], ',')[19] == "true" # sidecar overlap setting was honored
    end
end

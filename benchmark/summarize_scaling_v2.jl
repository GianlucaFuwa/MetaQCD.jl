using DelimitedFiles
using Plots
using Statistics
using TOML
include("hunter_roofline.jl")
using .HunterRoofline
using Plots.Measures
using Plots: plot, plot!, savefig

# Parse units explicitly: these are reported GFLOP/s and effective GB/s, not
# hardware counter measurements. Keep the user's maximum-throughput default.
function parse_benchmark_file(filepath; statistic=:max)
    column = get(Dict(:max => 1, :median => 2, :mean => 3), statistic, 0)
    column > 0 || error("statistic must be :max, :median, or :mean")
    results = Dict{String,Dict{String,Vector{Float64}}}()
    current_test = nothing
    for raw in eachline(filepath)
        line = strip(raw)
        isempty(line) && continue
        if occursin(r"float(?:16|32|64)", line)
            current_test = line
            results[line] = Dict("gflops" => Float64[], "bandwidth" => Float64[])
        elseif current_test !== nothing &&
            (occursin("[GFLOPs]", line) || occursin("[GB/s]", line))
            value = tryparse(Float64, split(line)[column])
            isnothing(value) && error("Missing $statistic in $filepath: $line")
            metric = occursin("[GFLOPs]", line) ? "gflops" : "bandwidth"
            push!(results[current_test][metric], value / 1000)
        end
    end
    return results
end

extract_nprocs_from_filename(filename) = HunterRoofline.benchmark_geometry(filename).nprocs
function get_scaling_type(filename)
    return if contains(filename, "strong")
        "strong"
    elseif contains(filename, "weak")
        "weak"
    else
        "unknown"
    end
end

function collect_scaling_data(directory_path; statistic=:max, mode="barrier")
    datasets = (Dict(), Dict())
    for filename in sort(readdir(directory_path))
        endswith(filename, ".txt") && !endswith(filename, "build.txt") || continue
        scaling = get_scaling_type(filename)
        scaling == "unknown" && continue
        path = joinpath(directory_path, filename)
        # New runs carry exact geometry/preferences; legacy files encode the grid.
        metadata_path = replace(path, r"\.txt$" => ".toml")
        metadata = isfile(metadata_path) ? TOML.parsefile(metadata_path) : Dict()
        geometry = if haskey(metadata, "numprocs_cart") && haskey(metadata, "global_dims")
            grid, dims = Tuple(metadata["numprocs_cart"]), Tuple(metadata["global_dims"])
            (; global_dims=dims, numprocs_cart=grid, nprocs=prod(grid))
        else
            HunterRoofline.benchmark_geometry(filename)
        end
        target = datasets[scaling == "strong" ? 1 : 2]
        for (raw_name, result) in parse_benchmark_file(path; statistic)
            endswith(raw_name, "[$mode]") || continue
            name = strip(replace(raw_name, r"\s*\[(?:no)?barrier\]$" => ""))
            all(length(result[key]) == 1 for key in ("gflops", "bandwidth")) ||
                error("Incomplete results for $raw_name in $path")
            data = get!(target, name) do
                Dict(
                    "nprocs" => Int[],
                    "gflops" => Float64[],
                    "bandwidth" => Float64[],
                    "geometry" => Any[],
                    "gauge_floats" => Int[],
                    "overlap" => Union{Nothing,Bool}[],
                    "refresh_halo" => Bool[],
                    "halo_width" => Int[],
                )
            end
            geometry.nprocs in data["nprocs"] &&
                error("Duplicate $name at $(geometry.nprocs) APUs")
            push!(data["nprocs"], geometry.nprocs)
            push!(data["gflops"], only(result["gflops"]))
            push!(data["bandwidth"], only(result["bandwidth"]))
            push!(data["geometry"], geometry)
            push!(data["gauge_floats"], get(metadata, "gauge_floats", 12))
            push!(data["overlap"], get(metadata, "hide_communication", nothing))
            push!(data["refresh_halo"], get(metadata, "refresh_halo", true))
            push!(data["halo_width"], get(metadata, "halo_width", 1))
        end
    end
    for dataset in datasets, data in values(dataset)
        order = sortperm(data["nprocs"])
        for key in keys(data)
            data[key] = data[key][order]
        end
    end
    return datasets
end

function collect_build_scaling_data(directory_path)
    datasets = (Matrix{Float64}[], Matrix{Float64}[])
    for filename in sort(readdir(directory_path))
        endswith(filename, "build.txt") || continue
        scaling = get_scaling_type(filename)
        scaling == "unknown" && continue
        push!(
            datasets[scaling == "strong" ? 1 : 2],
            readdlm(joinpath(directory_path, filename), Float64),
        )
    end
    return map(datasets) do tables
        isempty(tables) && return zeros(0, 2)
        combined = vcat(tables...)
        combined[sortperm(combined[:, 1]), :]
    end
end

function create_build_plots(strong_build_scaling_data, weak_build_scaling_data, output_dir)
    xscale, markersize = :identity, 7
    colors, markers = [:blue], [:circle]
    if !isempty(strong_build_scaling_data) && !isempty(weak_build_scaling_data)
        ngpus = strong_build_scaling_data[:, 1]
        pstrong = plot(
            strong_build_scaling_data[:, 1],
            1 ./ strong_build_scaling_data[:, 2];
            title="Strong Scaling - Multiple Walkers (TTS)",
            xlabel="Number of GPUs",
            ylabel="1 / TTS [1/s]",
            xticks=[2^i for i in 1:7],
            xscale=xscale,
            # yscale=:log10,
            legend=:topleft,
            grid=true,
            color=colors[1],
            marker=markers[1],
            linewidth=2,
            markersize=markersize,
            titlefontsize=12,
            label="Data",
        )
        min_gpus, idxmin = findmin(strong_build_scaling_data[:, 1])
        max_gpus, idxmax = findmax(strong_build_scaling_data[:, 1])
        base_perf = strong_build_scaling_data[idxmin, 2]
        ideal_perf = (ngpus ./ min_gpus) ./ base_perf

        plot!(
            pstrong,
            ngpus,
            ideal_perf;
            label="Ideal",
            color=colors[1],
            linestyle=:dash,
            alpha=0.7,
        )
        ngpus = weak_build_scaling_data[:, 1]
        pweak = plot(
            weak_build_scaling_data[:, 1],
            ngpus ./ weak_build_scaling_data[:, 2];
            title="Weak Scaling - Multiple Walkers (TTS)",
            xlabel="Number of GPUs",
            ylabel="",
            xticks=[2^i for i in 1:7],
            xscale=xscale,
            # yscale=:log10,
            legend=:topleft,
            grid=true,
            color=colors[1],
            marker=markers[1],
            markersize=markersize,
            linewidth=2,
            titlefontsize=12,
            label="Data",
        )
        min_gpus, idxmin = findmin(weak_build_scaling_data[:, 1])
        max_gpus, idxmax = findmax(weak_build_scaling_data[:, 1])
        base_perf = weak_build_scaling_data[idxmin, 2]
        ideal_perf = ngpus ./ base_perf

        plot!(
            pweak,
            ngpus,
            ideal_perf;
            label="Ideal",
            color=colors[1],
            linestyle=:dash,
            alpha=0.7,
        )
        plt = plot(
            pstrong,
            pweak;
            layout=grid(1, 2),
            size=(800, 350),
            bottom_margin=3mm,
            left_margin=5mm,
        )
        savefig(plt, joinpath(output_dir, "scaling_multiplewalkers.pdf"))
    else
        if !isempty(strong_build_scaling_data)
            ngpus = strong_build_scaling_data[:, 1]
            pstrong = plot(
                strong_build_scaling_data[:, 1],
                1 ./ strong_build_scaling_data[:, 2];
                title="Strong Scaling - Multiple Walkers (TTS)",
                xlabel="Number of GPUs",
                ylabel="1 / TTS [1/s]",
                xticks=[2^i for i in 1:7],
                xscale=xscale,
                # yscale=:log10,
                legend=:topleft,
                grid=true,
                color=colors[1],
                marker=markers[1],
                linewidth=2,
                markersize=markersize,
                label="Data",
                size=(550, 400),
            )
            min_gpus, idxmin = findmin(strong_build_scaling_data[:, 1])
            max_gpus, idxmax = findmax(strong_build_scaling_data[:, 1])
            base_perf = strong_build_scaling_data[idxmin, 2]
            ideal_perf = (ngpus ./ min_gpus) ./ base_perf

            plot!(
                pstrong,
                ngpus,
                ideal_perf;
                label="Ideal",
                color=colors[1],
                linestyle=:dash,
                alpha=0.7,
            )
            savefig(pstrong, joinpath(output_dir, "strong_scaling_multiplewalkers.pdf"))
        end

        if !isempty(weak_build_scaling_data)
            ngpus = weak_build_scaling_data[:, 1]
            pweak = plot(
                weak_build_scaling_data[:, 1],
                ngpus ./ weak_build_scaling_data[:, 2];
                title="Weak Scaling - Multiple Walkers (TTS)",
                xlabel="Number of GPUs",
                ylabel="1 / TTS [1/s]",
                xticks=[2^i for i in 1:7],
                xscale=xscale,
                # yscale=:log10,
                legend=:topleft,
                grid=true,
                color=colors[1],
                marker=markers[1],
                markersize=markersize,
                linewidth=2,
                label="Data",
                size=(550, 400),
            )
            min_gpus, idxmin = findmin(weak_build_scaling_data[:, 1])
            max_gpus, idxmax = findmax(weak_build_scaling_data[:, 1])
            base_perf = weak_build_scaling_data[idxmin, 2]
            ideal_perf = ngpus ./ base_perf

            plot!(
                pweak,
                ngpus,
                ideal_perf;
                label="Ideal",
                color=colors[1],
                linestyle=:dash,
                alpha=0.7,
            )
            savefig(pweak, joinpath(output_dir, "weak_scaling_multiplewalkers.pdf"))
        end
    end

    return nothing
end

function create_scaling_plots(
    strong_scaling_data,
    weak_scaling_data,
    strong_build_scaling_data,
    weak_build_scaling_data;
    output_dir="plots",
    hardware=HunterHardware(),
    overlap=nothing,
    hbm_traffic=:benchmark,
    placement=:contiguous,
    xscale=:log2,
)
    mkpath(output_dir)
    create_build_plots(strong_build_scaling_data, weak_build_scaling_data, output_dir)
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray]
    markers = [:circle, :square, :diamond, :utriangle, :dtriangle, :star5, :plus, :x]
    plots = []
    bandwidth_plots = []
    open(joinpath(output_dir, "scaling_model.csv"), "w") do io
        println(
            io,
            "scaling,test,apus,local_x,local_y,local_z,local_t,measured_tflops,measured_effective_tb_s,model_tflops,model_effective_tb_s,arithmetic_intensity,model_memory_ms,model_network_ms,model_compute_ms,send_mb_per_apu,max_remote_mb_per_node,boundary_fraction,overlap,hbm_traffic,placement",
        )
        for (scaling, dataset) in
            (("strong", strong_scaling_data), ("weak", weak_scaling_data))
            names = sort(filter(name -> !contains(name, "Copy"), collect(keys(dataset))))
            isempty(names) && continue
            ticks = sort(unique(vcat([dataset[name]["nprocs"] for name in names]...)))
            pperf = plot(;
                title="$(titlecase(scaling)) scaling — Performance",
                xlabel="Number of APUs",
                ylabel="Reported performance [TFLOP/s]",
                xticks=ticks,
                xscale=xscale,
                size=(600, 400),
                legend=:topleft,
                margin=4mm,
            )
            pbw = plot(;
                title="$(titlecase(scaling)) scaling — Throughput",
                xlabel="Number of APUs",
                ylabel="Effective throughput [TB/s]",
                xticks=ticks,
                xscale=xscale,
                size=(600, 400),
                legend=:topleft,
                margin=4mm,
            )
            model_added = false
            for (i, name) in enumerate(names)
                data = dataset[name]
                color, marker = colors[mod1(i, length(colors))],
                markers[mod1(i, length(markers))]
                ns = data["nprocs"]
                isempty(ns) && continue
                for (p, metric) in ((pperf, "gflops"), (pbw, "bandwidth"))
                    plot!(p, ns, data[metric]; label=name, color, marker, linewidth=2)
                    plot!(
                        p,
                        ns,
                        data[metric][1] .* ns ./ ns[1];
                        label="",
                        color,
                        linestyle=:dash,
                        alpha=0.4,
                    )
                end
                # if HunterRoofline.operator_work(name) !== nothing
                if false
                    run_overlap = [
                        isnothing(overlap) ? something(value, false) : overlap for
                        value in data["overlap"]
                    ]
                    ceilings = [
                        HunterRoofline.model_ceiling(
                            name,
                            geometry;
                            hardware,
                            overlap=run_overlap[j],
                            hbm_traffic,
                            placement,
                            gauge_floats=data["gauge_floats"][j],
                            refresh_halo=data["refresh_halo"][j],
                            halo_width=data["halo_width"][j],
                        ) for (j, geometry) in enumerate(data["geometry"])
                    ]
                    plot!(
                        pperf,
                        ns,
                        getproperty.(ceilings, :tflops);
                        color,
                        linestyle=:dot,
                        linewidth=2,
                        label="",
                    )
                    plot!(
                        pbw,
                        ns,
                        getproperty.(ceilings, :effective_tb_s);
                        color,
                        linestyle=:dot,
                        linewidth=2,
                        label="",
                    )
                    model_added = true
                    for (j, ceiling) in enumerate(ceilings)
                        row = (
                            scaling,
                            name,
                            ns[j],
                            ceiling.local_dims...,
                            data["gflops"][j],
                            data["bandwidth"][j],
                            ceiling.tflops,
                            ceiling.effective_tb_s,
                            ceiling.intensity,
                            1e3ceiling.memory_seconds,
                            1e3ceiling.net_seconds,
                            1e3ceiling.compute_seconds,
                            ceiling.sent_bytes_per_rank / 1e6,
                            ceiling.max_node_remote_bytes / 1e6,
                            ceiling.boundary_fraction,
                            run_overlap[j],
                            hbm_traffic,
                            placement,
                        )
                        println(io, join(row, ','))
                    end
                else
                    @warn "No hardware model for $name; omitting its ceiling"
                end
            end
            for p in (pperf, pbw)
                plot!(
                    p,
                    [NaN],
                    [NaN];
                    color=:black,
                    linestyle=:dash,
                    label="Ideal from first point",
                    alpha=0.4,
                )
                if model_added
                    plot!(
                        p,
                        [NaN],
                        [NaN];
                        color=:black,
                        linestyle=:dot,
                        linewidth=2,
                        label="Model max (HBM + network)",
                    )
                end
            end
            savefig(pperf, joinpath(output_dir, "$(scaling)_scaling_performance.pdf"))
            savefig(pbw, joinpath(output_dir, "$(scaling)_scaling_bandwidth.pdf"))
            push!(plots, pperf, pbw)
            push!(bandwidth_plots, pbw)
        end
    end
    if length(bandwidth_plots) == 2
        combined = plot(bandwidth_plots...; layout=(1, 2), size=(1100, 400))
        savefig(combined, joinpath(output_dir, "scaling_bandwidth.pdf"))
    end
    println("Plots and scaling_model.csv saved to $output_dir/")
    return plots
end

function print_scaling_summary(
    strong_scaling_data,
    weak_scaling_data,
    strong_build_scaling_data,
    weak_build_scaling_data,
)
    """Print a summary of scaling efficiency for both strong and weak scaling."""

    println("\nStrong Scaling Summary:")
    println("="^60)
    @show strong_build_scaling_data

    if !isempty(strong_build_scaling_data)
        perf = strong_build_scaling_data[:, 2]
        max_gpus = maximum(strong_build_scaling_data[:, 1])
        min_gpus = minimum(strong_build_scaling_data[:, 1])
        # For strong scaling: efficiency = (actual speedup) / (ideal speedup)
        actual_speedup_perf = perf[1] / perf[end]
        ideal_speedup = max_gpus / min_gpus
        perf_efficiency = (actual_speedup_perf / ideal_speedup) * 100
        println("Multiple Walkers:")
        println(
            "  Performance efficiency: $(round(perf_efficiency, digits=1))% (speedup: $(round(actual_speedup_perf, digits=2))x)",
        )
        println()
    end

    for (test_name, data) in strong_scaling_data
        if contains(test_name, "Copy") ||
            isempty(data["nprocs"]) ||
            length(data["nprocs"]) < 2
            continue
        end

        nprocs = data["nprocs"]
        gflops = data["gflops"]
        bandwidth = data["bandwidth"]

        # Calculate strong scaling efficiency (should increase linearly)
        if length(nprocs) >= 2
            max_procs = maximum(nprocs)
            min_procs = minimum(nprocs)
            if min_procs > 0 && max_procs > min_procs
                # For strong scaling: efficiency = (actual speedup) / (ideal speedup)
                actual_speedup_perf = gflops[end] / gflops[1]
                ideal_speedup = max_procs / min_procs
                perf_efficiency = (actual_speedup_perf / ideal_speedup) * 100

                actual_speedup_bw = bandwidth[end] / bandwidth[1]
                bw_efficiency = (actual_speedup_bw / ideal_speedup) * 100

                println("$test_name:")
                println(
                    "  Performance efficiency: $(round(perf_efficiency, digits=1))% (speedup: $(round(actual_speedup_perf, digits=2))x)",
                )
                println(
                    "  Throughput efficiency: $(round(bw_efficiency, digits=1))% (speedup: $(round(actual_speedup_bw, digits=2))x)",
                )
                println()
            end
        end
    end

    println("\nWeak Scaling Summary:")
    println("="^60)

    if !isempty(weak_build_scaling_data)
        perf = weak_build_scaling_data[:, 2]
        max_gpus = maximum(weak_build_scaling_data[:, 1])
        min_gpus = minimum(weak_build_scaling_data[:, 1])
        # For strong scaling: efficiency = (actual speedup) / (ideal speedup)
        actual_speedup_perf = perf[1] * max_gpus / perf[end]
        ideal_speedup = max_gpus / min_gpus
        perf_efficiency = (actual_speedup_perf / ideal_speedup) * 100
        println("Multiple Walkers:")
        println(
            "  Performance efficiency: $(round(perf_efficiency, digits=1))% (speedup: $(round(actual_speedup_perf, digits=2))x)",
        )
        println()
    end

    for (test_name, data) in weak_scaling_data
        if contains(test_name, "Copy") ||
            isempty(data["nprocs"]) ||
            length(data["nprocs"]) < 2
            continue
        end

        nprocs = data["nprocs"]
        gflops = data["gflops"]
        bandwidth = data["bandwidth"]

        # Calculate weak scaling efficiency (should remain constant)
        if length(nprocs) >= 2
            max_procs = maximum(nprocs)
            min_procs = minimum(nprocs)
            if min_procs > 0 && max_procs > min_procs
                # For weak scaling: efficiency = (performance at N procs) / (performance at 1 proc)
                actual_speedup_perf = gflops[end] / gflops[1]
                ideal_speedup = max_procs / min_procs
                perf_efficiency = (actual_speedup_perf / ideal_speedup) * 100

                actual_speedup_bw = bandwidth[end] / bandwidth[1]
                bw_efficiency = (actual_speedup_bw / ideal_speedup) * 100

                println("$test_name:")
                println("  Performance retention: $(round(perf_efficiency, digits=1))%")
                println("  Throughput retention: $(round(bw_efficiency, digits=1))%")
                println()
            end
        end
    end
end

# Main execution
function main(;
    benchmark_dir=joinpath(@__DIR__, "scaling_results_hunter3"),
    output_dir="plots",
    statistic=:max,
    overlap=nothing,
    hardware=HunterRoofline.HunterHardware(),
    hbm_traffic=:benchmark,
    placement=:contiguous,
    xscale=:identity,
)
    # Set the directory containing benchmark files

    println("Parsing benchmark files from: $benchmark_dir")
    strong_scaling_data, weak_scaling_data = collect_scaling_data(benchmark_dir; statistic)
    strong_build_scaling_data, weak_build_scaling_data = collect_build_scaling_data(
        benchmark_dir
    )

    if isempty(strong_scaling_data)
        if isempty(weak_scaling_data)
            println(
                "No benchmark data found. Please check the directory path and file format."
            )
            return nothing
        else
            println("Found $(length(weak_scaling_data)) weak scaling test types")
        end
    else
        println("Found $(length(strong_scaling_data)) strong scaling test types")
        if !isempty(weak_scaling_data)
            println("Found $(length(weak_scaling_data)) weak scaling test types")
        end
    end

    if !isempty(strong_build_scaling_data)
        println(
            "Found $(length(strong_build_scaling_data)) strong build scaling test types"
        )
    end

    if !isempty(weak_build_scaling_data)
        println("Found $(length(weak_build_scaling_data)) weak build scaling test types")
    end

    # Create plots
    plots = create_scaling_plots(
        strong_scaling_data,
        weak_scaling_data,
        strong_build_scaling_data,
        weak_build_scaling_data;
        output_dir,
        hardware,
        overlap,
        hbm_traffic,
        placement,
        xscale,
    )

    # Print summary
    print_scaling_summary(
        strong_scaling_data,
        weak_scaling_data,
        strong_build_scaling_data,
        weak_build_scaling_data,
    )

    # Display plots (optional - comment out if running in headless mode)
    for plot in plots
        # display(plot)
    end

    return strong_scaling_data,
    weak_scaling_data, strong_build_scaling_data,
    weak_build_scaling_data
end

# Run the main function
# if abspath(PROGRAM_FILE) == @__FILE__
main()
# end

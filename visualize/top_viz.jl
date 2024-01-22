using ColorSchemes
using DelimitedFiles
using MetaQCD
using Plots

# U = identity_gauges(16, 16, 16, 16, 1.25, DBW2GaugeAction)
# loadU!(BridgeFormat(), U, "configs/L16x16x16x16_dbw2_beta1.25/config_00000001.txt")
# qₙ = zeros(Float64, size(U))
# for site in eachindex(U)
#     global qₙ
#     qₙ[site] = MetaQCD.Measurements.top_charge_density_clover(U, site)
# end

function visualize_tc(qₙ)
    cmap = ColorSchemes.RdYlBu_8
    x, y, z = axes(qₙ)[1:3]

    fig = Figure(resolution = (1000, 900), fontsize=28)
    ax = Axis3(fig[1, 1], perspectiveness=0.3, azimuth=7.1, elevation=0.57,
               aspect=(1, 1, 1), xlabeloffset=60, ylabeloffset=60, zlabeloffset=60)
    ts = 2
    obj = contour!(ax, x, y, z, qₙ[:, :, :, ts];
            colormap=Reverse(cmap), alpha=1, #colorrange=(minimum(qₙ), maximum(qₙ)),
            levels = [collect(-1:0.01:-0.2)..., collect(0.2:0.01:1)...])
    Colorbar(fig[1, 2], obj; label="Topological charge density", height=Relative(0.5))
    fig
end

function animate_tc(qₙ; cmap=ColorSchemes.RdYlBu_8, alpha=0.1, clip=0.15, framerate=4)
    time = Observable(1)
    x, y, z = axes(qₙ)[1:3]

    fig = Figure(resolution = (1000, 900), fontsize=28)
    ax = Axis3(fig[1, 1], perspectiveness=0.3, azimuth=7.1, elevation=0.57,
               aspect=(1, 1, 1), xlabeloffset=60, ylabeloffset=60, zlabeloffset=60)
    density = @lift qₙ[:, :, :, $time]
    obj = contour!(ax, x, y, z, density;
            colormap=Reverse(cmap), alpha=alpha, colorrange=(minimum(qₙ), maximum(qₙ)),
            levels = [collect(-1:0.01:-clip)..., collect(clip:0.01:1)...])
    # Colorbar(fig[1, 2], obj; label="Topological charge density", height=Relative(0.5))

    record(fig, "top_charge_density.mp4", axes(qₙ)[4]; framerate=framerate) do t
        time[] = t
    end
end

with_theme(theme_dark()) do
    visualize_tc(qₙ)
end

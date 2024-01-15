include("MetaViz.jl")

measdir(name) = "master\\L16x16x16x16_dbw2_beta1.25\\$(name)\\raw_data"
biasfile(name) = "master\\L16x16x16x16_dbw2_beta1.25\\$(name)\\"

# timeseries(MetaMeasurements(measdir("opes30_2")), :bias_data_1)


biaspotential(MetaBias(biasfile("opes30_2"), which=:opes))

using CairoMakie
using LsqFit

colors = Makie.wong_colors()

bias_metad = MetaBias(biasfile("metad"), which=:metad)
bias_opes  = MetaBias(biasfile("opes30_2"), which=:opes)

meas_metad = MetaMeasurements(measdir("metad"))
meas_opes  = MetaMeasurements(measdir("opes30_2"))
meas_conv  = MetaMeasurements(measdir("conventional"))
meas_param = MetaMeasurements(measdir("param"))

function ts(meas1, meas2)
    itvl = 1:70000
    # itrj = getproperty(meas1, :topological_charge_flowed)["itrj"][itvl]
    # CV1 = getproperty(meas1, :topological_charge_flowed)["Qclover (tf=3.6)"][itvl]
    # CV2 = getproperty(meas2, :topological_charge_flowed)["Qclover (tf=3.6)"][itvl]
    itrj = getproperty(meas1, :bias_data_1)["itrj"][itvl]
    CV1 = getproperty(meas1, :bias_data_1)["cv"][itvl]
    CV2 = getproperty(meas2, :bias_data_1)["cv"][itvl]

    fig = Figure(resolution=(800, 350), fontsize=20)
    ax = Axis(fig[1, 1], xlabel="HMC Trajectories", xticks=(0:2e4:6e4),
              ylabel=L"Q_{\textrm{meta}}^{\textrm{SU(3)}}", yticks=-4:1:1,
              xtickalign=1, ytickalign=1, xgridvisible=false, ylabelsize=24)

    lines!(ax, itrj, CV2, label="OPES", color=colors[3])
    lines!(ax, itrj, CV1, label="MetaD", color=colors[5])
    axislegend(ax, position=:lb, framevisible=false)
    fig
end

fig = ts(meas_metad, meas_opes)

# name = raw"D:\UNI\Master-thesis\Figures\SU3_build_ts.pdf"
# save(name, fig)

# fig

model(x, p) = p[1]*cos.(2π*p[2]*x) .- p[1]

function bp(bias)
    xrange = -5:0.01:4.99
    yn = bias.(xrange)
    _, ys = analyze(yn, 100, robust=true)
    # chisq(p, d) = sum((d .- model()))
    ssa = ys[:, 1] .- maximum(ys[:, 1])
    bfit = curve_fit(model, xrange, ssa, [7, 1.18])
    @show coef(bfit)

    fig = Figure(resolution=(500, 400), fontsize=20)
    ax = Axis(fig[1, 1], ylabel=L"V_{\textrm{meta}}(Q)", xticks=-5:5,
              xlabel=L"Q_{\textrm{meta}}^{\textrm{SU(3)}}",
              xtickalign=1, ytickalign=1, xgridvisible=false, ygridvisible=false)

    lines!(ax, xrange, yn .- minimum(yn), label="Original", linewidth=2)
    lines!(ax, xrange, ssa, label="SSA", linewidth=2)
    lines!(ax, xrange, model(xrange, coef(bfit)), label="Fit", linewidth=2, linestyle=:dash)
    axislegend(ax, position=:lt, framevisible=false, labelsize=16)
    fig
end

fig = bp(bias_opes)

# name = raw"D:\UNI\Master-thesis\Figures\SU3_pot_ssa.pdf"
# save(name, fig)

# fig

# lines(getproperty(meas_param, :bias_data_1)["cv"])

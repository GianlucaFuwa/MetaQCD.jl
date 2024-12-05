function modify_bias(
    b; orders=[1], cvlims=b.bias.cvlims, L=80, savedir="", biasfactor=Inf, do_fit=false
)
    x = cvlims[1]:0.01:cvlims[2]
    yn = b.bias.(x)
    yt, ys = SingularSpectrumAnalysis.analyze(yn, L, robust=true)
    y = zeros(length(yt))
    fac = (1 + biasfactor) / biasfactor

    for i in axes(ys, 2)
        y .+= view(ys, :, i) * fac
    end
    y .-= minimum(y)

    plt = plot(x, yn, label="Raw (γ=$(biasfactor))", linewidth=2)
    plot!(plt, x, y, label="SSA", linewidth=2)

    if savedir != ""
        bin_width = round(b.bias.bin_width, sigdigits=3)
        xrange = cvlims[1]:bin_width:cvlims[2]
        yrange = y[1:Int64(div(bin_width, 0.01)):length(y)]
        savefile = joinpath(savedir, b.ensemblename * "_ssa" * b.ext)
        println("Saving SSA in file: $(savefile)")
        open(savefile, "w") do fp
            println(fp, "$(rpad("CV", 7))\t$(rpad("V(CV)", 7))")

            for i in eachindex(xrange)
                println(fp, "$(rpad(xrange[i], 7, "0"))\t$(rpad(yrange[i], 7, "0"))")
            end
        end
    end

    if do_fit
        for order in orders
            p0 = zeros(2order+1)
            p0 .= 1.0
            p0[1] = 5.0
            model = barrier_func(order)
            bfit = curve_fit(model, x, y, p0)
            fity = model(x, coef(bfit))

            if savedir != ""
                savefile_fit = joinpath(savedir, b.ensemblename * "_ssafit$(order)" * b.ext)
                println("Saving fit in file: $(savefile_fit)")
                open(savefile_fit, "w") do fp
                    println(fp, "$(rpad("CV", 7))\t$(rpad("V(CV)", 7))")

                    for i in eachindex(y)
                        println(fp, "$(rpad(x[i], 7, "0"))\t$(rpad(fity[i], 7, "0"))")
                    end
                end
            end

            plot!(plt, x, fity, label="SSA+Fit (order=$(order))", linewidth=2)
        end
    end

    display(plt)
    return Dict(
        "quadratic part" => yt,
        "sinusoidal part" => ys,
    )
end

function barrier_func(order)
    @assert order >= 1

    function f(x, p)
        out = p[1] .+ p[2] * cospi.(p[3]*x).^2

        for i in 2:order
            out .+= p[2(i-1)+2] * cospi.(p[2(i-1)+3]*x).^(2order)
        end

        return out
    end

    return f
end

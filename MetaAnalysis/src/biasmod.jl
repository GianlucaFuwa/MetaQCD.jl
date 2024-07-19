function modify_bias(b::MetaBias, order; savefile="", cvlims=b.bias.cvlims, L=80)
    x = cvlims[1]:0.01:cvlims[2]
    yn = b.bias.(x)
    yt, ys = SingularSpectrumAnalysis.analyze(yn, L, robust=true)
    # TODO: Fit and return parameters
    p0 = zeros(2order+1)
    p0 .= 1.0
    p0[1] = 5.0
    y = zeros(length(yt))

    for i in axes(ys, 2)
        y .+= view(ys, :, i)
    end

    y .-= minimum(y)

    if savefile != ""
        open(savefile, "w") do fp
            println(fp, "$(rpad("CV", 7))\t$(rpad("V(CV)", 7))")
            for i in eachindex(y)
                println(fp, "$(rpad(x[i], 7, "0"))\t$(rpad(y[i], 7, "0"))")
            end
        end
    end

    model = barrier_func(order)
    bfit = curve_fit(model, x, y, p0)
    plt = plot(x, y)
    fity = model(x, coef(bfit))
    plot!(plt, x, fity)
    display(plt)
    return (yt, ys), coef(bfit), sum(bfit.resid.^2)
end

function barrier_func(order)
    @assert order >= 1
    function f(x, p)
        out = p[1] .+ p[2] * cospi.(2p[3]*x)
        for i in 2:order
            out .+= p[2(i-1)+2] * cospi.(2p[2(i-1)+3]*x)
        end
        return out
    end
end

barrier_func(x, p) = p[1] * cospi.(p[2] * x).^2

function modify_bias(b::MetaBias; cvlims=b.bias.cvlims, L=80)
    x = cvlims[1]:0.01:cvlims[2]
    yn = b.bias.(x)
    yt, ys = SingularSpectrumAnalysis.analyze(yn, L, robust=true)
    # TODO: Fit and return parameters
    p0 = [3.0, 1.0]
    y = ys[:, 1] .- minimum(view(ys, :, 1))
    bfit = curve_fit(barrier_func, x, y, p0)
    # display(plot(x, yt))
    plt = plot(x, y)
    plot!(plt, x, xx -> barrier_func(xx, coef(bfit)))
    display(plt)
    return coef(bfit)
end

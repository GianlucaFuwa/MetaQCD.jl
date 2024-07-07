# from hep-lat/1401.3270
const SQRTT0 = Dict{Int64,uwreal}(
    0 => uwreal([0.1638, 0.0010], "t0"),
    2 => uwreal([0.1539, 0.0012], "t0"),
    3 => uwreal([0.1465, 0.0025], "t0"),
    4 => uwreal([0.1420, 0.0008], "t0"),
)

const W0 = Dict{Int64,uwreal}(
    0 => uwreal([0.1670, 0.0010], "w0"),
    2 => uwreal([0.1760, 0.0013], "w0"),
    3 => uwreal([0.1755, 0.0018], "w0"),
    4 => uwreal([0.1715, 0.0009], "w0"),
)

fm⁻¹_to_GeV(x) = x / (1/0.197)

function t0_scale(filename, Nf; save_filename="")
    dat = readdlm(filename, skipstart=1)
    firstitrj = dat[1, 1]
    flow_num = findfirst(x -> x!=firstitrj, view(dat, :, 1)) - 1
    flow_stepsize = dat[2, 3] - dat[1, 3]
    flow_times = range(flow_stepsize, flow_num * flow_stepsize; step=flow_stepsize)
    len = size(dat, 1)

    t²E = Vector{uwreal}(undef, flow_num)

    for (i, tf) in enumerate(flow_times)
        results = analyze(
            view(dat, i:flow_num:len-flow_num*20, 4),
            UWerr(),
            save_filename=save_filename,
        )
        E = uwreal([results["mean"], results["stderr"]], "$tf")
        # E = uwreal([mean(view(dat, i:flow_num:len, 4)), std(view(dat, i:flow_num:len, 4))], 1)
        uwerr(E)
        t²E[i] = tf^2 * E
        uwerr(t²E[i])
    end

    errs = ADerrors.err.(t²E)

    ℰ= Spline1D(flow_times, value.(t²E); w=1 ./ errs.^2, k=3)
    ℰ_err = Spline1D(flow_times, errs; k=3)
    W(t) = t * Dierckx.derivative(ℰ, t)
    W_err(t) = t * Dierckx.derivative(ℰ_err, t)

    t0_val = find_zero(t -> ℰ(t) - 0.3, (flow_times[1], flow_times[end]))
    t0_err = ℰ_err(t0_val)
    t0 = uwreal([t0_val, t0_err], "t0"); uwerr(t0)
    w0_val = sqrt(find_zero(t -> W(t) - 0.3, 2))
    w0_err = W_err(w0_val)
    w0 = uwreal([w0_val, w0_err], "w0"); uwerr(w0)
    if w0_val^2 > flow_times[end]
        @warn("w₀² is bigger than tf_max, so probably not reliable")
    end

    a_t0 = SQRTT0[Nf] / sqrt(t0); uwerr(a_t0)
    ainv_t0 = fm⁻¹_to_GeV(1/a_t0); uwerr(ainv_t0)
    a_w0 = W0[Nf] / w0; uwerr(a_w0)
    ainv_w0 = fm⁻¹_to_GeV(1/a_w0); uwerr(ainv_w0)

    if save_filename != ""
        io = open(save_filename, "a")
        println(io, "t0/a²: $(t0)")
        println(io, "a: $(phys_not(a_t0)) fm")
        println(io, "1/a: $(phys_not(ainv_t0))")
        close(io)
        # println("t0/a²: $(t0)")
        # println("a: $(phys_not(a)) fm")
        # println("1/a: $(phys_not(ainv))")
    end

    println("\nt₀ = $(phys_not(t0))")
    println("a from t₀ = $(phys_not(a_t0)) fm")
    println("a⁻¹ from t₀ = $(phys_not(ainv_t0)) GeV\n")
    println("w₀ = $(phys_not(w0))")
    println("a from w₀ = $(phys_not(a_w0)) fm")
    println("a⁻¹ from w₀ = $(phys_not(ainv_w0)) GeV")
    return flow_times, t²E, ℰ, x->W(x)
end

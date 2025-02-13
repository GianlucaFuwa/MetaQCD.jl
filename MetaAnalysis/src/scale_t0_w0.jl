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

# TODO: Determine error on t-function via bootstrap
function t0_scale(
    data, error_est; Nf=-1, save_filename="", Nt::Int64=-1,
)
    @assert 0 <= Nf <= 4 "Make sure Nf is set!"
    firstitrj = data[1, 1]
    flow_num = findfirst(x -> x!=firstitrj, view(data, :, 1)) - 1
    flow_stepsize = data[2, 3] - data[1, 3]
    flow_times = range(flow_stepsize, flow_num * flow_stepsize; step=flow_stepsize)
    len = size(data, 1)
    t²E = Vector{uwreal}(undef, flow_num)

    for (i, tf) in enumerate(flow_times)
        results = analyze(
            data[i:flow_num:len-flow_num, 4],
            error_est,
            save_filename=save_filename,
        )
        E = uwreal([results["mean"], results["stderr"]], "$tf")
        # E = uwreal([mean(view(dat, i:flow_num:len, 4)), std(view(dat, i:flow_num:len, 4))], 1)
        uwerr(E)
        t²E[i] = tf^2 * E
        uwerr(t²E[i])
    end

    errs = ADerrors.err.(t²E)

    ℰ = Spline1D(flow_times, value.(t²E); w=1 ./ errs.^2, k=3, bc="extrapolate")
    ℰ_err = Spline1D(flow_times, errs; k=3, bc="extrapolate")
    W(t) = t * Dierckx.derivative(ℰ, t)
    W_err(t) = t * Dierckx.derivative(ℰ_err, t)
    plt = plot(flow_times, t -> ℰ(t) - 0.3, ribbon=(t -> ℰ_err(t)))
    display(plt)

    t0_val = try
        find_zero(t -> ℰ(t) - 0.3, (flow_times[1], flow_times[end]))
    catch _
        error("Could not find t0, i.e., tf^2*E is not equal to 0.3 up to the maximum flow time")
    end
    t0_val_left = try
        find_zero(t -> ℰ(t) - 0.3 + ℰ_err(t), (flow_times[1], flow_times[end]))
    catch _
        error("Could not find t0, i.e., tf^2*E is not equal to 0.3 up to the maximum flow time")
    end
    t0_val_right = try
        find_zero(t -> ℰ(t) - 0.3 - ℰ_err(t), (flow_times[1], flow_times[end]*2))
    catch _
        error("Could not find t0, i.e., tf^2*E is not equal to 0.3 up to the maximum flow time")
    end

    t0_err = 0.5 * (abs(t0_val_left - t0_val) + abs(t0_val_right - t0_val))
    t0 = uwreal([t0_val, t0_err], "t0"); uwerr(t0)
    a_t0 = SQRTT0[Nf] / sqrt(t0); uwerr(a_t0)
    ainv_t0 = fm⁻¹_to_GeV(1/a_t0); uwerr(ainv_t0)
    T_t0 = ainv_t0 * 1000 / Nt; uwerr(T_t0)

    println("\nt₀ = $(phys_not(t0))")
    println("a from t₀ = $(phys_not(a_t0)) fm")
    println("a⁻¹ from t₀ = $(phys_not(ainv_t0)) GeV")
    Nt > 0 ? println("T from t₀, given Nt=$Nt = $(phys_not(T_t0)) MeV\n") : println()

    w0_val = try
        sqrt(find_zero(t -> W(t) - 0.3, 2))
    catch _
        error("Could not find w0^2")
    end
    w0_val_left = try
        sqrt(find_zero(t -> W(t) - 0.3 + W_err(t), 2))
    catch _
        error("Could not find w0^2")
    end
    w0_val_right = try
        sqrt(find_zero(t -> W(t) - 0.3 - W_err(t), 2))
    catch _
        @warn "w0 right sided error could not be determined; using left sided instead"
        w0_val_left
    end

    w0_err = 0.5 * (abs(w0_val_left - w0_val) + abs(w0_val_right - w0_val))
    w0 = uwreal([w0_val, w0_err], "w0"); uwerr(w0)
    a_w0 = W0[Nf] / w0; uwerr(a_w0)
    ainv_w0 = fm⁻¹_to_GeV(1/a_w0); uwerr(ainv_w0)
    T_w0 = ainv_w0 * 1000 / Nt; uwerr(T_w0)

    if w0_val^2 > flow_times[end]
        @warn("w0^2 is bigger than tf_max, so probably not reliable")
    end

    println("w₀ = $(phys_not(w0))")
    println("a from w₀ = $(phys_not(a_w0)) fm")
    println("a⁻¹ from w₀ = $(phys_not(ainv_w0)) GeV")
    Nt > 0 ? println("T from w₀, given Nt=$Nt = $(phys_not(T_w0)) MeV\n") : println()

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

    return Dict(
        "flow times" => flow_times,
        "tf^2E" => t²E,
        "T spline" => ℰ,
        "W spline" => x->W(x),
    )
end

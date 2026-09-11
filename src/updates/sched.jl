"""
    optimal_schedule(bias; q0, q1, ease=:cosine, ζfloor=1e-6, nfine=1001, fd_step=nothing)

Build a continuous CV-space switching schedule z(t), t ∈ [0,1], for steering a
constrained/steered-HMC move from q0 to q1 through a bias potential supplied as a
callable `bias(q) -> V(q)` (e.g. minus the accumulated metadynamics free-energy estimate).

Construction (two stages):

1.  BULK SHAPE — a cheap local-dissipation proxy ζ(q) ≈ |V''(q)| is estimated by
    finite-differencing `bias` on a fine grid between q0 and q1, and the path is
    reparametrized to move at constant speed in the metric √ζ(q) dq (crawling through
    high-curvature regions, moving fast through flat ones) — approximating the
    thermodynamic-geometry optimal protocol ζ(q) q̇² = const.

2.  BOUNDARY CLAMP — the metric-optimal parametrization is composed with an outer
    reparametrization h(t) satisfying h(0)=0, h(1)=1, h'(0)=h'(1)=0. This is required
    (not optional) for the reversibility of the RATTLE-type dynamics with a
    time-dependent constraint (Schönle et al., arXiv:2512.16812, Eq. II.6); omitting
    it produces measurable sampling bias (their Appendix H2).

Arguments
- bias     : callable, bias(q) -> Real. Only pointwise evaluations are used, so this
             can be a closure over a sum of Gaussians (metadynamics), a spline, a
             neural-net bias, etc. — no tabulation required.
- q0, q1   : start/end CV values of the move
- ease     : :cosine  -> h(t) = (1 - cos(pi*t))/2            [C^1 clamp]
             :quintic -> h(t) = t^3*(10 - 15t + 6t^2)         [C^2 clamp, "minimum jerk"]
- ζfloor   : minimum allowed friction-proxy value (guards against V'' ≈ 0 at
             inflection points, which would otherwise let the schedule move
             arbitrarily fast there)
- nfine    : resolution of the internal finite-difference / arclength / inversion grid
             (this many evaluations of `bias`, times 3, are done once at construction)
- fd_step  : finite-difference step for estimating V''(q); defaults to the fine-grid
             spacing |q1-q0|/(nfine-1). Increase this if `bias` is noisy (e.g. a raw,
             un-smoothed metadynamics estimate), decrease it if `bias` is very sharp.

Returns a NamedTuple (z, dzdt):
- z(t)     : the position schedule, callable at any t ∈ [0,1] (not restricted to a grid)
- dzdt(t)  : the derivative dz/dt. Rescale by 1/(K*Δt), where K is the number of
             integration steps you use and Δt the MD timestep, to obtain the physical
             velocity schedule v_z(t_k) required by the RATTLE update (paper's Eq. II.10).
             dzdt(0) = dzdt(1) = 0 by construction.

All the construction cost (finite differences + arclength integration) happens once,
up front; z(t) and dzdt(t) themselves are cheap O(log nfine) table lookups per call.
"""
function optimal_schedule(bias, q0, q1; ease=:cosine, ζfloor=1e-6, nfine=1001, fd_step=nothing)
    @assert q0 != q1 "q0 and q1 must differ"
    @assert nfine >= 3 "nfine must be at least 3"

    h_fd = fd_step === nothing ? max(abs(q1 - q0)/(nfine - 1), 1e-8) : Float64(fd_step)

    # --- fine grid between q0 and q1 (works for q1 < q0 too) ---
    qfine = collect(range(q0, q1; length=nfine))

    # --- curvature proxy ζ(q) ≈ |V''(q)| via central finite differences of `bias` ---
    Vpp = [ (bias(x + h_fd) - 2*bias(x) + bias(x - h_fd)) / h_fd^2 for x in qfine ]
    ζ   = max.(abs.(Vpp), ζfloor)
    g   = sqrt.(ζ)   # = sqrt(ζ) tabulated on qfine

    # --- signed metric-arclength integral u(q) via trapezoid rule ---
    u = zeros(nfine)
    for j in 2:nfine
        u[j] = u[j-1] + 0.5*(g[j-1] + g[j])*(qfine[j] - qfine[j-1])
    end
    utot = u[end]
    @assert utot != 0 "degenerate metric distance between q0 and q1"
    ubar = u ./ utot   # monotonically increasing in [0,1], regardless of sign(q1-q0)

    # --- sqrt(ζ) interpolated off the same fine-grid table used for construction ---
    function sqrtζ_of_q(x::Real)
        if x <= qfine[1];   return g[1];   end
        if x >= qfine[end]; return g[end]; end
        j = clamp(searchsortedlast(qfine, x), 1, nfine-1)
        t = (x - qfine[j]) / (qfine[j+1] - qfine[j])
        return (1-t)*g[j] + t*g[j+1]
    end

    # --- inverse map: normalized metric progress p in [0,1]  ->  q ---
    function invert_progress(p::Real)
        p = clamp(p, 0.0, 1.0)
        j = clamp(searchsortedlast(ubar, p), 1, nfine-1)
        t = (p - ubar[j]) / (ubar[j+1] - ubar[j])
        return (1-t)*qfine[j] + t*qfine[j+1]
    end

    # --- outer, boundary-clamped reparametrization h(t): h(0)=0, h(1)=1, h'(0)=h'(1)=0 ---
    if ease == :cosine
        hfun  = t -> (1 - cos(pi*t))/2
        hpfun = t -> pi*sin(pi*t)/2
    elseif ease == :quintic
        hfun  = t -> t^3*(10 - 15t + 6t^2)
        hpfun = t -> 30*t^2*(1-t)^2
    else
        error("unknown ease type $ease (use :cosine or :quintic)")
    end

    # --- assemble the two callables ---
    function z(t::Real)
        tt = clamp(t, 0.0, 1.0)
        return invert_progress(hfun(tt))
    end

    function dzdt(t::Real)
        tt = clamp(t, 0.0, 1.0)
        zt = z(tt)
        # dz/dt = dz/dp * dp/dt = (dq/du) * h'(t) = (utot / sqrtζ(z)) * h'(t)
        return utot / sqrtζ_of_q(zt) * hpfun(tt)
    end

    return (z=z, dzdt=dzdt)
end


# ------------------------- usage example -------------------------
# bias = q -> -log(0.5*exp(-20*(q-1)^2) + 0.5*exp(-20*(q+1)^2))   # e.g. -FES of a double well
# sched = optimal_schedule(bias; q0=-1.0, q1=1.0)
# sched.z(0.37)        # position at normalized time t=0.37, evaluated on demand
# sched.dzdt(0.0)      # == 0.0 exactly, by construction
#
# # sampling K+1 points for an actual RATTLE trajectory with K steps and timestep Δt:
# K, Δt = 100, 1e-3
# tk  = (0:K) ./ K
# zk  = sched.z.(tk)
# vzk = sched.dzdt.(tk) ./ (K*Δt)

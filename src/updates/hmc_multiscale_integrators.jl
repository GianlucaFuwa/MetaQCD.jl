# =============================================================================
# Multirate HMC integrator with per-level independent step sizes
# =============================================================================
#
# Design
# ------
# Each level has its own integrator type and step size Δτ.  The integrator is
# built at macro-expansion time by:
#
#   1. Unrolling each level's kick-drift (KD) sequence for (τ_lcm / Δτ_lvl)
#      steps, merging palindromic boundary kicks between consecutive steps.
#   2. Merging all per-level unrolled streams via a pending-drift algorithm
#      that respects the fixed intra-step ordering (negative drifts, e.g. the
#      δ step in OMF4, are handled correctly because we never re-sort events
#      within a stream — only inter-stream drift races are resolved).
#   3. Wrapping the merged LCM-period block in a loop of n_lcm_repeats, where
#      n_lcm_repeats = numsteps_coarsest / (τ_lcm / Δτ_coarsest).
#   4. Emitting a flat, straight-line evolve!() body with literal Float64
#      coefficients — zero runtime overhead beyond the force evaluations.
#
# numsteps semantics
# ------------------
# numsteps belongs to the COARSEST level (largest Δτ).  It is the number of
# coarse steps in the full trajectory, so the trajectory length is
#     τ = numsteps_coarsest × Δτ_coarsest
# The LCM block is repeated  numsteps_coarsest / (τ_lcm / Δτ_coarsest)  times.
# numsteps_coarsest must be a multiple of (τ_lcm / Δτ_coarsest); a compile-time
# error is raised otherwise (or you can use the runtime-checked version below).
#
# Coefficient convention
# ----------------------
# updateP!(U, hmc, coeff, ..., level) multiplies coeff by hmc.levels[level].Δτ.
# updateU!(U, hmc, coeff, ..., level) multiplies coeff by hmc.levels[level].Δτ.
# The generated code emits literal absolute values divided by the runtime Δτ,
# preserving the existing contract with one division per call (negligible vs
# force evaluations).
#
# Usage
# -----
#   @multirate_integrator MyInt (
#       level=1, integrator=OMF4, Δτ=0.2
#   ) (
#       level=2, integrator=OMF2, Δτ=0.15
#   )
#
#   # Then use exactly like any other integrator:
#   evolve!(MyInt(), U, hmc, fermion_action, bias, therm, 1)
#
# Inspect the schedule without defining a struct:
#   events, τ_lcm = multirate_schedule(
#       (level=1, integrator=OMF4, Δτ=0.2),
#       (level=2, integrator=OMF2, Δτ=0.15),
#   )


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Kick-drift sequence descriptors
#
# Each function returns the KD sequence for ONE step of the integrator as a
# Vector of (Symbol, Float64): (:k, fraction) or (:d, fraction).
# The sequence starts and ends with :k.  Drift fractions sum to 1, kick
# fractions sum to 1.  Negative drift fractions (e.g. OMF4's δ) are allowed.
# ─────────────────────────────────────────────────────────────────────────────

function kd_sequence(int::String)
    seq = if lowercase(int) == "leapfrog"
        kd_sequence(Leapfrog)
    elseif lowercase(int) == "omf2"
        kd_sequence(OMF2)
    elseif lowercase(int) == "omf4"
        kd_sequence(OMF4)
    end

    return seq
end

function kd_sequence(::Type{Leapfrog})
    return [(:k, 0.5), (:d, 1.0), (:k, 0.5)]
end

function kd_sequence(::Type{OMF2})
    α = 0.1931833275037836
    γ = 1.0 - 2.0 * α
    return [(:k, α), (:d, 0.5), (:k, γ), (:d, 0.5), (:k, α)]
end

function kd_sequence(::Type{OMF4})
    α = 0.08398315262876693
    β = 0.2539785108410595
    γ = 0.6822365335719091
    δ = -0.03230286765269967
    μ = 0.5 - γ - α
    ν = 1.0 - 2δ - 2β
    return [(:k,α), (:d,β), (:k,γ), (:d,δ),
            (:k,μ), (:d,ν), (:k,μ), (:d,δ),
            (:k,γ), (:d,β), (:k,α)]
end


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Unroll n_steps of a KD sequence, merging palindromic boundary kicks
# ─────────────────────────────────────────────────────────────────────────────

"""
    unroll_steps(kd_seq, n_steps, Δτ) -> Vector{Tuple{Symbol,Float64}}

Repeat `kd_seq` for `n_steps`, merging the end-kick of step i with the
start-kick of step i+1.  All values are scaled by Δτ.
"""
function unroll_steps(kd_seq, n_steps, Δτ::Float64)
    out = Tuple{Symbol,Float64}[]
    for _ in 1:n_steps
        scaled = [(kind, frac * Δτ) for (kind, frac) in kd_seq]
        if !isempty(out) && out[end][1] == :k && scaled[1][1] == :k
            # Merge boundary kicks
            out[end] = (:k, out[end][2] + scaled[1][2])
            append!(out, scaled[2:end])
        else
            append!(out, scaled)
        end
    end
    return out
end


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Merge two level streams via pending-drift interleaving
# ─────────────────────────────────────────────────────────────────────────────

"""
    merge_two_streams(s1, s2, lvl1, lvl2) -> Vector

Merge two unrolled KD streams into a single flat event sequence.
Returns Vector of either (:d, abs_drift::Float64) or
(:k, abs_kick::Float64, level::Int).

The algorithm maintains a "pending drift" for each stream and emits
min(pending1, pending2) at each step, then fires whichever stream(s)
reach zero.  Negative drifts (OMF4's δ) are handled correctly because
events within each stream are always consumed in their original order.
"""
function merge_two_streams(s1::Vector, s2::Vector, lvl1::Int, lvl2::Int)
    result = Any[]
    i1, i2 = 1, 1

    function advance_kicks!(s, i, lvl)
        while i <= length(s) && s[i][1] == :k
            push!(result, (:k, s[i][2], lvl))
            i += 1
        end
        return i
    end

    # Both streams start with t=0 kicks
    i1 = advance_kicks!(s1, i1, lvl1)
    i2 = advance_kicks!(s2, i2, lvl2)

    # Load first drift from each stream
    pending1 = i1 <= length(s1) ? s1[i1][2] : nothing;  i1 += (i1 <= length(s1) ? 1 : 0)
    pending2 = i2 <= length(s2) ? s2[i2][2] : nothing;  i2 += (i2 <= length(s2) ? 1 : 0)

    while pending1 !== nothing || pending2 !== nothing
        dt = if     pending1 === nothing; pending2
             elseif pending2 === nothing; pending1
             else min(pending1, pending2) end

        push!(result, (:d, dt))

        if pending1 !== nothing
            pending1 = pending1 - dt
            if abs(pending1) < 1e-13; pending1 = 0.0; end
        end
        if pending2 !== nothing
            pending2 = pending2 - dt
            if abs(pending2) < 1e-13; pending2 = 0.0; end
        end

        if pending1 === 0.0
            i1 = advance_kicks!(s1, i1, lvl1)
            if i1 <= length(s1)
                pending1 = s1[i1][2]; i1 += 1
            else
                pending1 = nothing
            end
        end

        if pending2 === 0.0
            i2 = advance_kicks!(s2, i2, lvl2)
            if i2 <= length(s2)
                pending2 = s2[i2][2]; i2 += 1
            else
                pending2 = nothing
            end
        end
    end

    return result
end

"""
    merge_all_streams(streams) -> Vector

Merge an arbitrary number of (stream, level) pairs by sequential pairwise
merging.  For two levels this is exact; for three or more the pairwise order
doesn't affect correctness (associativity of the merge).
"""
function merge_all_streams(streams::Vector)
    # streams: Vector of (unrolled_events, level_int)
    if length(streams) == 1
        s, lvl = streams[1]
        # Tag single stream
        return [ev[1] == :k ? (:k, ev[2], lvl) : (:d, ev[2]) for ev in s]
    end
    # Merge pairwise left-to-right
    s1, lvl1 = streams[1]
    merged = merge_two_streams(s1, streams[2][1], lvl1, streams[2][2])
    for k in 3:length(streams)
        sk, lvlk = streams[k]
        # Wrap merged result as a "stream" with a sentinel level (already tagged)
        # Re-merge: treat merged as stream of tagged events, sk as new stream
        merged = merge_tagged_with_stream(merged, sk, lvlk)
    end
    return merged
end

"""
    merge_tagged_with_stream(tagged, s2, lvl2)

Merge an already-tagged event list (output of a previous merge) with a new
unrolled stream.  The tagged stream's drift values drive the pending-drift
logic; kicks are emitted as-is.
"""
function merge_tagged_with_stream(tagged::Vector, s2::Vector, lvl2::Int)
    result = Any[]
    i1, i2 = 1, 1

    # Advance past t=0 kicks in tagged stream
    while i1 <= length(tagged) && tagged[i1][1] == :k
        push!(result, tagged[i1])
        i1 += 1
    end
    # Advance past t=0 kicks in s2
    while i2 <= length(s2) && s2[i2][1] == :k
        push!(result, (:k, s2[i2][2], lvl2))
        i2 += 1
    end

    pending1 = i1 <= length(tagged) ? tagged[i1][2] : nothing; i1 += (i1 <= length(tagged) ? 1 : 0)
    pending2 = i2 <= length(s2)     ? s2[i2][2]    : nothing; i2 += (i2 <= length(s2) ? 1 : 0)

    while pending1 !== nothing || pending2 !== nothing
        dt = if     pending1 === nothing; pending2
             elseif pending2 === nothing; pending1
             else min(pending1, pending2) end

        push!(result, (:d, dt))

        if pending1 !== nothing
            pending1 = pending1 - dt
            if abs(pending1) < 1e-13; pending1 = 0.0; end
        end
        if pending2 !== nothing
            pending2 = pending2 - dt
            if abs(pending2) < 1e-13; pending2 = 0.0; end
        end

        if pending1 === 0.0
            while i1 <= length(tagged) && tagged[i1][1] == :k
                push!(result, tagged[i1]); i1 += 1
            end
            if i1 <= length(tagged)
                pending1 = tagged[i1][2]; i1 += 1
            else
                pending1 = nothing
            end
        end

        if pending2 === 0.0
            while i2 <= length(s2) && s2[i2][1] == :k
                push!(result, (:k, s2[i2][2], lvl2)); i2 += 1
            end
            if i2 <= length(s2)
                pending2 = s2[i2][2]; i2 += 1
            else
                pending2 = nothing
            end
        end
    end

    return result
end


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — LCM and numsteps bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

function lcm_rational(a::Rational{Int}, b::Rational{Int})
    return lcm(a.num * b.den, b.num * a.den) // (a.den * b.den)
end

"""
    build_schedule(level_specs)

Given level specs (sorted by Δτ ascending), return:
  - `events`:       merged flat event list over one LCM period
  - `τ_lcm`:        LCM period as Float64
  - `n_lcm_per_coarsest_step`: how many LCM periods fit in one coarsest step
                    (= τ_lcm / Δτ_coarsest, always an integer)

The caller is responsible for wrapping the event block in a loop of
    numsteps_coarsest ÷ n_lcm_per_coarsest_step
repetitions (and asserting that numsteps is divisible).
"""
function build_schedule(level_specs)
    # Sort finest → coarsest
    specs = sort(collect(level_specs); by = s -> s.Δτ)

    # Rational step sizes
    Δτ_rats = [rationalize(s.Δτ; tol=1e-9) for s in specs]

    # LCM period
    τ_lcm_r = reduce(lcm_rational, Δτ_rats)
    τ_lcm   = Float64(τ_lcm_r)

    # Number of steps per level in one LCM period
    n_steps_per_lcm = [τ_lcm_r // Δτ for Δτ in Δτ_rats]
    @assert all(d.den == 1 for d in n_steps_per_lcm) "LCM computation failed"

    # How many coarsest steps fit in one LCM period
    coarsest_Δτ_r   = Δτ_rats[end]
    n_lcm_per_coarsest_step = coarsest_Δτ_r // τ_lcm_r  # ≤ 1 as Rational
    # Equivalently: the number of LCM blocks per trajectory is
    #   numsteps_coarsest * Δτ_coarsest / τ_lcm
    # which must be integer.  We store n_coarsest_steps_per_lcm = τ_lcm/Δτ_coarsest
    # so that the loop count = numsteps_coarsest / n_coarsest_steps_per_lcm.
    n_coarsest_steps_per_lcm = τ_lcm_r // coarsest_Δτ_r
    @assert n_coarsest_steps_per_lcm.den == 1
    n_coarsest_per_lcm = Int(n_coarsest_steps_per_lcm.num)

    # Unroll each level
    streams = []
    for (spec, n, Δτ_r) in zip(specs, n_steps_per_lcm, Δτ_rats)
        n_int = Int(n.num)
        Δτ_f  = Float64(Δτ_r)
        kd    = kd_sequence(spec.integrator)
        s     = unroll_steps(kd, n_int, Δτ_f)
        push!(streams, (s, spec.level))
    end

    # Merge all streams
    events = merge_all_streams(streams)

    return events, τ_lcm, n_coarsest_per_lcm, specs
end


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Code generation
# ─────────────────────────────────────────────────────────────────────────────

"""
    generate_evolve_body(events, specs, n_coarsest_per_lcm) -> Expr

Emit:
  - An assertion that numsteps_coarsest is divisible by n_coarsest_per_lcm
  - A for loop of (numsteps_coarsest ÷ n_coarsest_per_lcm) iterations
  - Inside: one updateU! or updateP! call per event, with literal Float64 coefficients
"""
function generate_evolve_body(events, specs, n_coarsest_per_lcm::Int, numlevels)
    coarsest_spec  = specs[end]   # largest Δτ
    coarsest_level = coarsest_spec.level
    finest_level   = specs[1].level  # updateU! always uses finest level

    inner = Expr[]
    n_ev  = length(events)
    kicks_per_level = zeros(Int64, numlevels)
    kicks_done = zeros(Int64, numlevels)

    for ev in events
        if ev[1] != :d
            lvl = ev[3]
            kicks_per_level[lvl] += 1
        end
    end

    for (idx, ev) in enumerate(events)
        if ev[1] == :d
            # updateU! coefficient = abs_drift / hmc.levels[finest_level].Δτ
            abs_drift = ev[2]
            call = quote
                updateU!(U, hmc, $(abs_drift) / hmc.levels[$(finest_level)].Δτ,
                         fermion_action, bias, therm, $(finest_level))
            end
            push!(inner, call)
        else  # :k
            abs_kick = ev[2]
            lvl      = ev[3]
            is_last = (kicks_done[lvl] == kicks_per_level[lvl]-1)
            # updateP! coefficient = abs_kick / hmc.levels[lvl].Δτ
            if is_last
                call = quote
                    updateP!(U, hmc, $(abs_kick) / hmc.levels[$(lvl)].Δτ,
                             fermion_action, bias, $(lvl), true)
                end
            else
                call = quote
                    updateP!(U, hmc, $(abs_kick) / hmc.levels[$(lvl)].Δτ,
                             fermion_action, bias, $(lvl))
                end
            end
            push!(inner, call)
            kicks_done[lvl] += 1
        end
    end

    # Loop: numsteps_coarsest must be divisible by n_coarsest_per_lcm
    body = quote
        let numsteps = hmc.levels[$(coarsest_level)].numsteps
            @assert numsteps % $(n_coarsest_per_lcm) == 0 string(
                "numsteps=", numsteps, " for level ", $(coarsest_level),
                " must be a multiple of ", $(n_coarsest_per_lcm),
                " (= τ_lcm/Δτ_coarsest) for this multirate integrator"
            )
            n_reps = numsteps ÷ $(n_coarsest_per_lcm)
            for _ in 1:n_reps
                $(inner...)
            end
        end
    end

    return body
end


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Macro
# ─────────────────────────────────────────────────────────────────────────────

"""
    @multirate_integrator Name spec1 spec2 ...

Define a zero-field struct `Name <: AbstractIntegrator` and its `evolve!` method
implementing a multirate HMC integrator with independent per-level step sizes.

Each spec is a named-tuple literal:
    (level = <Int>, integrator = <IntegratorType>, Δτ = <Number>)

`level`      — index into `hmc.levels`
`integrator` — one of Leapfrog, OMF2, OMF4 (extensible via kd_sequence)
`Δτ`         — step size for this level; must be rational (representable as p/q)

The coarsest level (largest Δτ) owns `numsteps`; the trajectory length is
    τ = numsteps_coarsest × Δτ_coarsest
and `numsteps_coarsest` must be a multiple of (τ_lcm / Δτ_coarsest).

Examples
--------
    # Two levels: OMF4 gauge + OMF2 fermion, non-divisible step sizes
    @multirate_integrator GaugeFermion2L (
        level=1, integrator=OMF4, Δτ=0.2
    ) (
        level=2, integrator=OMF2, Δτ=0.15
    )

    # Single level (equivalent to OMF4Slow, useful for debugging)
    @multirate_integrator DebugOMF4 (
        level=1, integrator=OMF4, Δτ=0.1
    )

    # Three levels
    @multirate_integrator ThreeLevel (
        level=1, integrator=OMF4, Δτ=0.1
    ) (
        level=2, integrator=OMF2, Δτ=0.3
    ) (
        level=3, integrator=Leapfrog, Δτ=0.6
    )
"""
macro multirate_integrator(name, specs...)
    # Parse specs from Expr(:tuple, ...) with named fields
    numlevels = length(specs)
    level_specs = map(specs) do spec
        d = Dict{Symbol,Any}()
        for arg in spec.args
            @assert Meta.isexpr(arg, :(=)) "Each field must be `key = value`, got: $arg"
            d[arg.args[1]] = arg.args[2]
        end
        (level      = d[:level]::Int,
         integrator = Core.eval(__module__, d[:integrator]),
         Δτ         = Float64(d[:Δτ]))
    end

    # Build schedule at macro-expansion time (compile time)
    # @show level_specs    
    events, τ_lcm, n_coarsest_per_lcm, specs_sorted = build_schedule(level_specs)
    # @show τ_lcm, n_coarsest_per_lcm

    # Generate the evolve! body
    body = generate_evolve_body(events, specs_sorted, n_coarsest_per_lcm, numlevels)

    # Build show string
    spec_strs = join(
        ["level=$(s.level) $(s.integrator)(Δτ=$(s.Δτ))" for s in specs_sorted],
        ", "
    )
    show_str = "$(name)[$(spec_strs), τ_lcm=$(τ_lcm), n_coarsest_per_lcm=$(n_coarsest_per_lcm)]"

    struct_def = quote
        struct $(esc(name)) <: AbstractIntegrator end
    end

    qualified_name = esc(GlobalRef(Updates, :evolve!))
    method_def = quote
        function $(qualified_name)(::$(esc(name)), U, hmc::HMC,
                         fermion_action, bias, therm, level)
            $body
            return nothing
        end
        function Base.show(io::IO, ::MIME"text/plain", ::$(esc(name)))
            print(io, $show_str)
        end
        Base.show(io::IO, ::$(esc(name))) = print(io, $show_str)
    end

    return Expr(:block, struct_def, method_def)
end

function build_multirate_integrator(specs, mod)
    # specs e.g. [(level=1, integrator=:OMF4, Δτ=0.2), (level=2, integrator=:OMF2, Δτ=0.15)]
    
    # Build a unique type name so repeated calls don't clash
    type_name = gensym("MultirateIntegrator")
    
    spec_exprs = [
        :($(s.level), $(s.integrator), $(s.Δτ))
        for s in specs
    ]
    
    # @eval runs the macro at global (module) scope
    @eval mod @multirate_integrator $type_name $(
        [:(level=$(s.level), integrator=$(s.integrator), Δτ=$(s.Δτ)) for s in specs]...
    )
    
    return @eval mod $type_name()  # return an instance
end

# ─────────────────────────────────────────────────────────────────────────────
# Convenience: inspect schedule without defining a struct
# ─────────────────────────────────────────────────────────────────────────────

"""
    multirate_schedule(specs...) -> (events, τ_lcm, n_coarsest_per_lcm)

Inspect the merged event schedule for a given set of level specs without
defining any struct.  Useful for debugging and verification.

Example
-------
    events, τ_lcm, n_c = multirate_schedule(
        (level=1, integrator=OMF4, Δτ=0.2),
        (level=2, integrator=OMF2, Δτ=0.15),
    )
    println("τ_lcm = ", τ_lcm)
    println("numsteps must be a multiple of ", n_c)
    for ev in events
        if ev[1] == :d
            @printf "  updateU!(%.10f)\\n" ev[2]
        else
            @printf "  updateP![lvl=%d](%.10f)\\n" ev[3] ev[2]
        end
    end
"""
function multirate_schedule(specs...)
    events, τ_lcm, n_coarsest_per_lcm, specs_sorted = build_schedule(collect(specs))
    return events, τ_lcm, n_coarsest_per_lcm
end

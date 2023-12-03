struct Heatbath{ITR,TOR,NHB,NOR} <: AbstractUpdate
    MAXIT::Int64
    numHB::Int64
    OR::TOR
    numOR::Int64
end

function Heatbath(::Gaugefield{GA}, eo, MAXIT, numHB, or_alg, numOR) where {GA}
    @level1("┌ Setting Heatbath...")

    if eo
        @level1("|  PARALLELIZATION ENABLED")
        @level1("|  Parallel heatbath doesn't always produce reproducible results.
        To force it, turn MT off with \"eo = false\" under [\"Physical Settings\"].")
        if GA==WilsonGaugeAction
            ITR = Checkerboard2MT
        else
            ITR = Checkerboard4MT
        end
    else
        @level1("|  PARALLELIZATION DISABLED")
        if GA==WilsonGaugeAction
            ITR = Checkerboard2
        else
            ITR = Checkerboard4
        end
    end
    @level1("|  ITERATOR: $(ITR)")

    OR = Overrelaxation(or_alg)
    TOR = typeof(OR)
    @level1("|  MAX. ITERATION COUNT IN HEATBATH: $(MAXIT)")
    @level1("|  NUM. OF HEATBATH SWEEPS: $(numHB)")
    @level1("|  OVERRELAXATION ALGORITHM: $(TOR)")
    @level1("|  NUM. OF OVERRELAXATION SWEEPS: $(numOR)")
    @level1("└\n")
    return Heatbath{ITR,TOR,Val{numHB},Val{numOR}}(MAXIT, numHB, OR, numOR)
end

function update!(hb::Heatbath{ITR,TOR,NHB,NOR}, U; kwargs...) where {ITR,TOR,NHB,NOR}
    numaccepts = 0.0

    sweep!(ITR(), NHB(), hb, U, U.NC/U.β)
    numaccepts += sweep_reduce!(ITR(), NOR(), TOR(), U, -U.β/U.NC)

    U.Sg = calc_gauge_action(U)
    numaccepts = (NOR≡Val{0}) ? 1.0 : numaccepts / (4*U.NV*_unwrap_val(NOR()))
    return numaccepts
end

function (hb::Heatbath)(U, μ, site, action_factor)
    old_link = U[μ][site]
    A = staple(U, μ, site)
    new_link = heatbath_SU3(old_link, A, hb.MAXIT, action_factor)
    U[μ][site] = new_link
    return nothing
end

function heatbath_SU3(old_link, A, MAXIT, action_factor)
    subblock = make_submatrix(cmatmul_od(old_link, A), 1, 2)
    tmp = embed_into_SU3(heatbath_SU2(subblock, MAXIT, action_factor), 1, 2)
    old_link = cmatmul_oo(tmp, old_link)

    subblock = make_submatrix(cmatmul_od(old_link, A), 1, 3)
    tmp = embed_into_SU3(heatbath_SU2(subblock, MAXIT, action_factor), 1, 3)
    old_link = cmatmul_oo(tmp, old_link)

    subblock = make_submatrix(cmatmul_od(old_link, A), 2, 3)
    tmp = embed_into_SU3(heatbath_SU2(subblock, MAXIT, action_factor), 2, 3)
    new_link = cmatmul_oo(tmp, old_link)
    return new_link
end

function heatbath_SU2(A, MAXIT, action_factor)
    r₀ = 1
    λ² = 1
    a_norm = 1 / sqrt(real(det(A)))
    V = a_norm * A
    i = 1

    while r₀^2 + λ² >= 1
        if i > MAXIT
            return eye2
        end

        r₁ = 1 - rand(Float64)
        x₁ = log(r₁)
        r₂ = 1 - rand(Float64)
        x₂ = cos(2π * r₂)
        r₃ = 1 - rand(Float64)
        x₃ = log(r₃)

        λ² = (-0.25 * action_factor * a_norm) * (x₁ + x₂^2*x₃)

        r₀ = rand(Float64)
        i += 1
    end

    x₀ = 1 - 2λ²
    abs_x = sqrt(1 - x₀^2)

    φ = rand(Float64)
    cosϑ = 1 - 2rand(Float64)
    vec_norm = abs_x * sqrt(1 - cosϑ^2)

    x₁ = vec_norm * cos(2π*φ)
    x₂ = vec_norm * sin(2π*φ)
    x₃ = abs_x * cosϑ

    mat = @SMatrix [
        x₀+im*x₃ x₂+im*x₁
        -x₂+im*x₁ x₀-im*x₃
    ]
    return cmatmul_od(mat, V)
end

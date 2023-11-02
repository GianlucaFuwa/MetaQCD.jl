struct Heatbath{ITR,TOR} <: AbstractUpdate
    MAXIT::Int64
    numHB::Int64
    OR::TOR
    numOR::Int64

    function Heatbath(
        ::Gaugefield{GA},
        eo, MAXIT, numHB, or_alg, numOR;
        verbose = nothing,
    ) where {GA}
        println_verbose1(verbose, ">> Setting Heatbath...")

        if eo
            println_verbose1(verbose, "\t>> PARALLELIZATION ENABLED")
            @info "Parallel heatbath doesn't always produce reproducible results. To force
            it, turn MT off with \"eo = false\" under [\"Physical Settings\"]."
            if GA==WilsonGaugeAction
                ITR = Checkerboard2MT
            else
                ITR = Checkerboard4MT
            end
        else
            println_verbose1(verbose, "\t>> PARALLELIZATION DISABLED")
            if GA==WilsonGaugeAction
                ITR = Checkerboard2
            else
                ITR = Checkerboard4
            end
        end
        println_verbose1(verbose, "\t>> ITERATOR = $(ITR)")

        OR = Overrelaxation(or_alg)
        TOR = typeof(OR)
        println_verbose1(verbose, "\t>> MAX. ITERATION COUNT IN HEATBATH = $(MAXIT)")
        println_verbose1(verbose, "\t>> NUM. OF HEATBATH SWEEPS = $(numHB)")
        println_verbose1(verbose, "\t>> OVERRELAXATION ALGORITHM = $(TOR)")
        println_verbose1(verbose, "\t>> NUM. OF OVERRELAXATION SWEEPS = $(numOR)\n")
        return new{ITR, TOR}(MAXIT, numHB, OR, numOR)
    end
end

function update!(hb::Heatbath{ITR}, U, ::VerboseLevel; kwargs...) where {ITR}
    numOR = hb.numOR
    numaccepts = 0.0

    sweep!(ITR(), hb.numHB, hb, U, U.NC/U.β)
    normalize!(U)
    numaccepts += sweep_reduce!(ITR(), hb.numOR, hb.OR, U, U.β/U.NC)
    normalize!(U)

    U.Sg = calc_gauge_action(U)
    numaccepts = (numOR==0) ? 1.0 : numaccepts / (4*U.NV*numOR)
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

# XXX: Maybe better to use a backup U so we dont have to bother with checkerboarding
struct Heatbath{ITR,TOR,NHB,NOR} <: AbstractUpdate
    MAXIT::Int64
    numHB::Int64
    OR::TOR
    numOR::Int64
end

function Heatbath(::Gaugefield{D,T,A,GA}, eo, MAXIT, numHB, or_alg, numOR) where {D,T,A,GA}
    @level1("┌ Setting Heatbath...")
    ITR = GA==WilsonGaugeAction ? Checkerboard2 : Checkerboard4
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
    GA = gauge_action(U)()
    @latmap(ITR(), NHB(), hb, U, GA, U.NC/U.β)
    numaccepts_or = @latsum(ITR(), NOR(), TOR(), U, GA, -U.β/U.NC)

    U.Sg = calc_gauge_action(U)
    numaccepts = (NOR≡Val{0}) ? 1.0 : numaccepts_or / (4*U.NV*_unwrap_val(NOR()))
    return numaccepts
end

function (hb::Heatbath)(U, μ, site, GA, action_factor)
    old_link = U[μ,site]
    A = staple(GA, U, μ, site)
    U[μ,site] = heatbath_SU3(old_link, A, hb.MAXIT, action_factor)
    return nothing
end

function heatbath_SU3(old_link::SMatrix{3,3,Complex{T},9}, A, MAXIT,
    action_factor) where {T}
    subblock = make_submatrix_12(cmatmul_od(old_link, A))
    tmp = embed_into_SU3_12(heatbath_SU2(subblock, MAXIT, action_factor))
    old_link = cmatmul_oo(tmp, old_link)

    subblock = make_submatrix_13(cmatmul_od(old_link, A))
    tmp = embed_into_SU3_13(heatbath_SU2(subblock, MAXIT, action_factor))
    old_link = cmatmul_oo(tmp, old_link)

    subblock = make_submatrix_23(cmatmul_od(old_link, A))
    tmp = embed_into_SU3_23(heatbath_SU2(subblock, MAXIT, action_factor))
    new_link = cmatmul_oo(tmp, old_link)
    return new_link
end

function heatbath_SU2(A::SMatrix{2,2,Complex{T},4}, MAXIT, action_factor) where {T}
    r₀ = 1
    λ² = 1
    a_norm = 1 / sqrt(real(det(A)))
    V = a_norm * A
    i = 1

    while r₀^2 + λ² >= 1
        if i > MAXIT
            return eye2(T)
        end

        r₁ = 1 - rand(T)
        x₁ = log(r₁)
        r₂ = 1 - rand(T)
        x₂ = cospi(2r₂)
        r₃ = 1 - rand(T)
        x₃ = log(r₃)

        λ² = (-T(0.25) * action_factor * a_norm) * (x₁ + x₂^2*x₃)

        r₀ = rand(T)
        i += 1
    end

    x₀ = 1 - 2λ²
    abs_x = sqrt(1 - x₀^2)

    φ = rand(T)
    cosϑ = 1 - 2rand(T)
    vec_norm = abs_x * sqrt(1 - cosϑ^2)

    x₁ = vec_norm * cospi(2φ)
    x₂ = vec_norm * sinpi(2φ)
    x₃ = abs_x * cosϑ

    mat = @SMatrix [
        x₀+im*x₃ x₂+im*x₁
        -x₂+im*x₁ x₀-im*x₃
    ]
    return cmatmul_od(mat, V)
end

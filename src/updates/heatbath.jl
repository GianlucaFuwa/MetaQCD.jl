"""
    Heatbath(U::Gaugefield{B,T,A,GA}, MAXIT, numHB, or_alg, numOR) where {B,T,A,GA}

Create a `Heatbath`` object.

# Arguments
- `U`: The gauge field on which the update is performed.
- `MAXIT`: The maximum iteration count in the Heatbath update.
- `numHB`: The number of Heatbath sweeps.
- `or_alg`: The overrelaxation algorithm used.
- `numOR`: The number of overrelaxation sweeps.

# Returns
A Heatbath object with the specified parameters. The gauge action `GA` of the field `U`
determines the iterator used. For the plaquette or Wilson action it uses a
Checkerboard iterator and for rectangular actions it partitions the lattice into four
sublattices.
"""
struct Heatbath{MAXIT,ITR,TOR,NHB,NOR} <: AbstractUpdate end

# @inline MAXIT(::Heatbath{MAXIT}) where {MAXIT} = _unwrap_val(MAXIT)
# @inline ITR(::Heatbath{<:Any,ITR}) where {ITR} = _unwrap_val(ITR)
# @inline TOR(::Heatbath{<:Any,<:Any,TOR}) where {TOR} = TOR
# @inline NHB(::Heatbath{<:Any,<:Any,<:Any,NHB}) where {NHB} = _unwrap_val(NHB)
# @inline NOR(::Heatbath{<:Any,<:Any,<:Any,<:Any,NOR}) where {NOR} = _unwrap_val(NOR)

function Heatbath(::Gaugefield{B,T,A,GA}, MAXIT, numHB, or_alg, numOR) where {B,T,A,GA}
    @level1("┌ Setting Heatbath...")
    ITR = GA == WilsonGaugeAction ? Checkerboard2 : Checkerboard4
    @level1("|  ITERATOR: $(ITR)")

    OR = Overrelaxation(or_alg)
    TOR = typeof(OR)
    @level1("|  MAX. ITERATION COUNT IN HEATBATH: $(MAXIT)")
    @level1("|  NUM. OF HEATBATH SWEEPS: $(numHB)")
    @level1("|  OVERRELAXATION ALGORITHM: $(TOR)")
    @level1("|  NUM. OF OVERRELAXATION SWEEPS: $(numOR)")
    @level1("└\n")
    return Heatbath{Val{MAXIT},ITR,TOR,Val{numHB},Val{numOR}}()
end

function update!(hb::Heatbath{<:Any,ITR,TOR,NHB,NOR}, U; kwargs...) where {ITR,TOR,NHB,NOR}
    GA = gauge_action(U)()
    @latmap(ITR(), NHB(), hb, U, GA, U.NC / U.β)
    numaccepts_or = @latsum(ITR(), NOR(), TOR(), U, GA, -U.β / U.NC)

    U.Sg = calc_gauge_action(U)
    numaccepts = (NOR ≡ Val{0}) ? 1.0 : numaccepts_or / (4 * U.NV * _unwrap_val(NOR()))
    return numaccepts
end

function (hb::Heatbath{MAXIT})(U, μ, site, GA, action_factor) where {MAXIT}
    old_link = U[μ, site]
    A = staple(GA, U, μ, site)
    U[μ, site] = heatbath_SU3(old_link, A, _unwrap_val(MAXIT()), action_factor)
    return nothing
end

function heatbath_SU3(old_link, A, MAXIT, action_factor)
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
    r₀ = one(T)
    λ² = one(T)
    a_norm = 1 / sqrt(real(det(A)))
    V = a_norm * A
    λ_factor = -1//4 * T(action_factor) * a_norm
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

        λ² = λ_factor * (x₁ + x₂^2 * x₃)

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

    mat = SMatrix{2,2,Complex{T},4}((x₀ + im*x₃, -x₂ + im*x₁, x₂ + im*x₁, x₀ - im*x₃))
    return cmatmul_od(mat, V)
end

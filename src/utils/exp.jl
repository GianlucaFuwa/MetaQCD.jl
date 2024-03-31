using Base.Math: isinf_real
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
# convenient struct to store exponential coefficients
struct exp_iQ_su3{T}
    Q::SU{3,9,T}
    Q²::SU{3,9,T}
    expiQ::SU{3,9,T}
    f₀::Complex{T}
    f₁::Complex{T}
    f₂::Complex{T}
    b₁₀::Complex{T}
    b₁₁::Complex{T}
    b₁₂::Complex{T}
    b₂₀::Complex{T}
    b₂₁::Complex{T}
    b₂₂::Complex{T}
end

function exp_iQ_su3(::Type{T}) where {T}
    Q = zero3(T)
    Q² = zero3(T)
    expiQ = zero3(T)
    T0 = zero(Complex{T})
    return exp_iQ_su3{T}(Q, Q², expiQ, T0, T0, T0, T0, T0, T0, T0, T0, T0)
end

Base.zero(::Type{exp_iQ_su3{T}}) where {T} = exp_iQ_su3(T)
exp_iQ(e::exp_iQ_su3{T}) where {T} = e.expiQ
get_Q(e::exp_iQ_su3{T}) where {T} = e.Q
get_Q²(e::exp_iQ_su3{T}) where {T} = e.Q²
get_B₁(e::exp_iQ_su3{T}) where {T} = e.b₁₀ * eye3(T) + e.b₁₁ * e.Q + e.b₁₂ * e.Q²
get_B₂(e::exp_iQ_su3{T}) where {T} = e.b₂₀ * eye3(T) + e.b₂₁ * e.Q + e.b₂₂ * e.Q²

"""
    exp_iQ(Q::SU{3,9,T}) where {T}
    exp_iQ(e::exp_iQ_su3{T}) where {T}

Compute the exponential of a traceless Hermitian 3x3 matrix `Q` or return the `exp_iQ` field
of the `exp_iQ_su3{T}`-object `e`. \\
From Morningstar & Peardon (2008) arXiv:hep-lat/0311018v1
"""
function exp_iQ(Q::SU{3,9,T}) where {T}
    u, w, signflip = set_uw(Q)
    f₀, f₁, f₂, _ = set_fj(T, u, w, signflip)
    mat = f₀*eye3(T) + f₁*Q + f₂*cmatmul_oo(Q, Q)
    return mat
end

"""
    exp_iQ_coeffs(Q::SU{3,9,T}) where {T}

Return a `exp_iQ_su3` object that contains the exponential of `Q` and all parameters
obtained in the Cayley-Hamilton algorithm that are needed for Stout force recursion.
"""
function exp_iQ_coeffs(Q::SU{3,9,T}) where {T}
    f₀, f₁, f₂, b₁₀, b₁₁, b₁₂, b₂₀, b₂₁, b₂₂ = calc_coefficients(Q)
    Q² = cmatmul_oo(Q, Q)
    mat = f₀*eye3(T) + f₁*Q + f₂*Q²
    return exp_iQ_su3(Q, Q², mat, f₀, f₁, f₂, b₁₀, b₁₁, b₁₂, b₂₀, b₂₁, b₂₂)
end

function calc_coefficients(Q::SU{3,9,T}) where {T}
    u, w, signflip = set_uw(Q)
    f₀, f₁, f₂, ξ₀ = set_fj(T, u, w, signflip)
    e²ⁱᵘ = cis(2u)
    e⁻ⁱᵘ = cis(-u)
    cosw = cos(w)
    w² = w*w
    u² = u*u

    if abs(w) <= T(0.2)
        ξ₁ = -T(1/3) + w²/30 * (one(T) - w²/28 * (one(T) - w²/54))
    else
        ξ₁ = cosw/w² - sin(w)/(w²*w)
    end

    r₁₀ = 2(u + im * (u² - w²)) * e²ⁱᵘ +
        2e⁻ⁱᵘ * (4u * (2 - im*u) * cosw +
        im * (9u² + w² - im*u * (3u² + w²)) * ξ₀)
    r₁₁ = 2(1 + 2im*u) * e²ⁱᵘ + e⁻ⁱᵘ * (-2(1 - im*u) * cosw +
        im * (6u + im * (w² - 3u²)) * ξ₀)
    r₁₂ = 2im*e²ⁱᵘ + im*e⁻ⁱᵘ * (cosw - 3(1 - im*u) * ξ₀)
    r₂₀ = -2*e²ⁱᵘ + 2im*u*e⁻ⁱᵘ * (cosw + (1 + 4im*u) * ξ₀ + 3u²*ξ₁)
    r₂₁ = -im*e⁻ⁱᵘ * (cosw + (1 + 2im*u) * ξ₀ - 3u²*ξ₁)
    r₂₂ = e⁻ⁱᵘ * (ξ₀ - 3im*u*ξ₁)

    bdenom = isapprox(zero(T), 9u² - w²) ? zero(T) : 1 / 2(9u² - w²)^2

    if signflip
        b₁₀ =  conj((2u*r₁₀ + (3u² - w²) * r₂₀ - 2(15u² + w²) *  conj(f₀))) * bdenom
        b₂₀ = -conj((r₁₀ - 3u*r₂₀ - 24u *  conj(f₀))) * bdenom
        b₁₁ = -conj((2u*r₁₁ + (3u² - w²) * r₂₁ - 2(15u² + w²) * -conj(f₁))) * bdenom
        b₂₁ =  conj((r₁₁ - 3u*r₂₁ - 24u * -conj(f₁))) * bdenom
        b₁₂ =  conj((2u*r₁₂ + (3u² - w²) * r₂₂ - 2(15u² + w²) *  conj(f₂))) * bdenom
        b₂₂ = -conj((r₁₂ - 3u*r₂₂ - 24u *  conj(f₂))) * bdenom
    else
        b₁₀ = (2u*r₁₀ + (3u² - w²) * r₂₀ - 2(15u² + w²) * f₀) * bdenom
        b₂₀ = (r₁₀ - 3u*r₂₀ - 24u*f₀) * bdenom
        b₁₁ = (2u*r₁₁ + (3u² - w²) * r₂₁ - 2(15u² + w²) * f₁) * bdenom
        b₂₁ = (r₁₁ - 3u*r₂₁ - 24u*f₁) * bdenom
        b₁₂ = (2u*r₁₂ + (3u² - w²) * r₂₂ - 2(15u² + w²) * f₂) * bdenom
        b₂₂ = (r₁₂ - 3u*r₂₂ - 24u*f₂) * bdenom
    end

    return f₀, f₁, f₂, b₁₀, b₁₁, b₁₂, b₂₀, b₂₁, b₂₂
end

function set_fj(::Type{T}, u, w, signflip) where {T}
    w² = w*w
    u² = u*u
    #if abs(w) <= 0.05
    #    w² = w²
    #    ξ₀ = 1 + 1/6 * w² * (1 - 1/20 * w² * (1 - 1/42 * w²))
    # end
    ξ₀ = iszero(w) ? one(w) : (isinf_real(w) ? zero(w) : sin(w)/(w))

    e²ⁱᵘ = cis(2u)
    e⁻ⁱᵘ = cis(-u)
    cosw = cos(w)
    denom = 9u² - w²

    if signflip
        if isapprox(zero(T), denom)
            fdenom = one(T)
            h₀ = one(Complex{T})
            h₁ = zero(Complex{T})
            h₂ = zero(Complex{T})
        else
            fdenom = one(T) / denom
            h₀ = conj((u² - w²) * e²ⁱᵘ + e⁻ⁱᵘ * (8u²*cosw + 2im*u * (3u² + w²) * ξ₀))
            h₁ = -conj(2u*e²ⁱᵘ - e⁻ⁱᵘ * (2u*cosw - im * (3u² - w²) * ξ₀))
            h₂ = conj(e²ⁱᵘ - e⁻ⁱᵘ * (cosw + 3im*u*ξ₀))
        end
    else
        if isapprox(zero(T), denom)
            fdenom = one(T)
            h₀ = one(Complex{T})
            h₁ = zero(Complex{T})
            h₂ = zero(Complex{T})
        else
            fdenom = one(T) / denom
            h₀ = (u² - w²) * e²ⁱᵘ + e⁻ⁱᵘ * (8u²*cosw + 2im*u * (3u² + w²) * ξ₀)
            h₁ = 2u*e²ⁱᵘ - e⁻ⁱᵘ * (2u*cosw - im * (3u² - w²) * ξ₀)
            h₂ = e²ⁱᵘ - e⁻ⁱᵘ * (cosw + 3im*u*ξ₀)
        end
    end

    f₀ = h₀ * fdenom
    f₁ = h₁ * fdenom
    f₂ = h₂ * fdenom
    return f₀, f₁, f₂, ξ₀
end

function set_uw(Q::SU{3,9,T}) where {T}
    oneover3 = T(1/3)
    c₀_bare = real(det(Q))
    signflip = c₀_bare < zero(T)
    c₀ = abs(c₀_bare)
    c₁ = T(0.5)*real(multr(Q, Q))
    c₁_3r = sqrt(c₁*oneover3)
    c₀ᵐᵃˣ = 2*(c₁_3r * c₁_3r * c₁_3r)
    Θ = isnan(c₀/c₀ᵐᵃˣ) ? acos(one(T)) : acos(min(one(T), c₀/c₀ᵐᵃˣ))

    u = c₁_3r * cos(Θ*oneover3)
    w = sqrt(c₁) * sin(Θ*oneover3)
    return u, w, signflip
end
end

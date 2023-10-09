# convenient struct to store exponential coefficients
struct exp_iQ_su3
    Q::SMatrix{3,3,ComplexF64,9}
    Q²::SMatrix{3,3,ComplexF64,9}
    expiQ::SMatrix{3,3,ComplexF64,9}
    f₀::ComplexF64
    f₁::ComplexF64
    f₂::ComplexF64
    b₁₀::ComplexF64
    b₁₁::ComplexF64
    b₁₂::ComplexF64
    b₂₀::ComplexF64
    b₂₁::ComplexF64
    b₂₂::ComplexF64
end

function exp_iQ_su3()
    Q = @SMatrix zeros(ComplexF64, 3, 3)
    Q² = @SMatrix zeros(ComplexF64, 3, 3)
    expiQ = @SMatrix zeros(ComplexF64, 3, 3)
    vals = @SVector zeros(ComplexF64, 9)
    return exp_iQ_su3(Q, Q², expiQ, vals...)
end

exp_iQ(e::exp_iQ_su3) = e.expiQ
get_Q(e::exp_iQ_su3) = e.Q
get_Q²(e::exp_iQ_su3) = e.Q²
get_B₁(e::exp_iQ_su3) = e.b₁₀ * eye3 + e.b₁₁ * e.Q + e.b₁₂ * e.Q²
get_B₂(e::exp_iQ_su3) = e.b₂₀ * eye3 + e.b₂₁ * e.Q + e.b₂₂ * e.Q²

"""
Exponential function of Q for traceless-hermitian Q \\
From Morningstar & Peardon (2008) arXiv:hep-lat/0311018v1
"""

function exp_iQ(Q::SMatrix{3,3,ComplexF64,9})
    u, w, signflip = set_uw(Q)
    f₀, f₁, f₂, _ = set_fj(u, w, signflip)
    mat = f₀ * eye3 + f₁ * Q + f₂ * cmatmul_oo(Q, Q)
    return mat
end

function exp_iQ_coeffs(Q::SMatrix{3,3,ComplexF64,9})
    f₀, f₁, f₂, b₁₀, b₁₁, b₁₂, b₂₀, b₂₁, b₂₂ = calc_coefficients(Q)
    Q² = cmatmul_oo(Q, Q)
    mat = f₀ * eye3 + f₁ * Q + f₂ * Q²
    return exp_iQ_su3(Q, Q², mat, f₀, f₁, f₂, b₁₀, b₁₁, b₁₂, b₂₀, b₂₁, b₂₂)
end

function calc_coefficients(Q::SMatrix{3,3,ComplexF64,9})
    u, w, signflip = set_uw(Q)
    f₀, f₁, f₂, ξ₀ = set_fj(u, w, signflip)
    e²ⁱᵘ = cis(2u)
    e⁻ⁱᵘ = cis(-u)
    cosw = cos(w)
    w² = w^2
    u² = u^2

    if abs(w) <= 1e-1
        ξ₁ = -1/3 + w²/30 * (1 - w²/28 * (1 - w²/54))
    else
        ξ₁ = cosw/w² - sin(w)/w^3
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

    bdenom = isapprox(0.0, 9u² - w²) ? 0.0 : 1 / 2(9u² - w²)^2

    if signflip
        b₁₀ = conj((2u*r₁₀ + (3u² - w²) * r₂₀ - 2(15u² + w²) * conj(f₀))) * bdenom
        b₂₀ = -conj((r₁₀ - 3u*r₂₀ - 24u*conj(f₀))) * bdenom
        b₁₁ = -conj((2u*r₁₁ + (3u² - w²) * r₂₁ - 2(15u² + w²) * -conj(f₁))) * bdenom
        b₂₁ = conj((r₁₁ - 3u*r₂₁ - 24u*conj(f₁))) * bdenom
        b₁₂ = conj((2u*r₁₂ + (3u² - w²) * r₂₂ - 2(15u² + w²) * -conj(f₂))) * bdenom
        b₂₂ = -conj((r₁₂ - 3u*r₂₂ - 24u*conj(f₂))) * bdenom
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

function set_fj(u, w, signflip)
    w² = w^2
    u² = u^2
    #if abs(w) <= 0.05
    #    w² = w²
    #    ξ₀ = 1 + 1/6 * w² * (1 - 1/20 * w² * (1 - 1/42 * w²))
    if isapprox(0.0, abs(w))
        ξ₀ = 1.0
    else
        ξ₀ = sin(w)/w
    end

    e²ⁱᵘ = cis(2u)
    e⁻ⁱᵘ = cis(-u)
    cosw = cos(w)
    denom = 9u² - w²

    if signflip
        if isapprox(0.0, denom)
            fdenom = 1.0
            h₀ = one(ComplexF64)
            h₁ = zero(ComplexF64)
            h₂ = zero(ComplexF64)
        else
            fdenom = 1 / denom
            h₀ = conj((u² - w²) * e²ⁱᵘ + e⁻ⁱᵘ * (8u²*cosw + 2im*u * (3u² + w²) * ξ₀))
            h₁ = -conj(2u*e²ⁱᵘ - e⁻ⁱᵘ * (2u*cosw - im * (3u² - w²) * ξ₀))
            h₂ = conj(e²ⁱᵘ - e⁻ⁱᵘ * (cosw + 3im*u*ξ₀))
        end
    else
        if isapprox(0.0, denom)
            fdenom = 1.0
            h₀ = one(ComplexF64)
            h₁ = zero(ComplexF64)
            h₂ = zero(ComplexF64)
        else
            fdenom = 1 / denom
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

function set_uw(Q::SMatrix{3,3,ComplexF64,9})
    c₀_bare = real(det(Q))
    signflip = c₀_bare < 0
    c₀ = abs(c₀_bare)
    c₁ = 0.5 * real(multr(Q, Q))
    c₁_3r = sqrt(c₁/3)
    c₀ᵐᵃˣ = 2 * c₁_3r^3
    Θ = isnan(c₀/c₀ᵐᵃˣ) ? acos(1.0) : acos(min(1.0, c₀/c₀ᵐᵃˣ))

    u = c₁_3r * cos(Θ/3)
    w = sqrt(c₁) * sin(Θ/3)
    return u, w, signflip
end

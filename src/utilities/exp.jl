# convenient struct to store exponential coefficients
struct exp_iQ_su3
    Q::SMatrix{3,3,ComplexF64,9}
    Q2::SMatrix{3,3,ComplexF64,9}
    expiQ::SMatrix{3,3,ComplexF64,9}
    f0::ComplexF64
    f1::ComplexF64
    f2::ComplexF64
    b10::ComplexF64
    b11::ComplexF64
    b12::ComplexF64
    b20::ComplexF64
    b21::ComplexF64
    b22::ComplexF64

    function exp_iQ_su3()
        Q = @SMatrix zeros(ComplexF64, 3, 3)
        Q2 = @SMatrix zeros(ComplexF64, 3, 3)
        expiQ = @SMatrix zeros(ComplexF64, 3, 3)
        vals = @SVector zeros(ComplexF64, 9)
        return new(Q, Q2, expiQ, vals...)
    end

    function exp_iQ_su3(Q, Q2, expiQ, f0, f1, f2, b10, b11, b12, b20, b21, b22)
        return new(Q, Q2, expiQ, f0, f1, f2, b10, b11, b12, b20, b21, b22)
    end

end

exp_iQ(e::exp_iQ_su3) = e.expiQ

B1(e::exp_iQ_su3) = e.b10 * eye3 + e.b11 * e.Q + e.b12 * e.Q2

B2(e::exp_iQ_su3) = e.b20 * eye3 + e.b21 * e.Q + e.b22 * e.Q2

"""
Exponential function of Q for traceless-hermitian Q \\
From Morningstar & Peardon (2008) arXiv:hep-lat/0311018v1
"""

function exp_iQ(Q::SMatrix{3,3,ComplexF64,9})
    u, w, signflip = set_uw(Q)
    f0, f1, f2, _ = set_fj(u, w, signflip)
    mat = f0 * eye3 + f1 * Q + f2 * cmatmul_oo(Q, Q)
    return mat
end

function exp_iQ_coeffs(Q::SMatrix{3,3,ComplexF64,9})
    f0, f1, f2, b10, b11, b12, b20, b21, b22 = calc_coefficients(Q)
    Q2 = cmatmul_oo(Q, Q)
    mat = f0 * eye3 + f1 * Q + f2 * Q2
    return exp_iQ_su3(Q, Q2, mat, f0, f1, f2, b10, b11, b12, b20, b21, b22)
end

function calc_coefficients(Q::SMatrix{3,3,ComplexF64,9})
    u, w, signflip = set_uw(Q)
    f0, f1, f2, ξ0 = set_fj(u, w, signflip)
    e2iu = cis(2u)
    emiu = cis(-u)
    cosw = cos(w)
    w2 = w^2
    u2 = u^2

    if abs(w) <= 1e-1
        ξ1 = -1/3 + 1/30 * w2 * (1 - 1/28 * w2 * (1 - 1/54 * w2))
    else
        ξ1 = cosw / w2 - sin(w) / w^3
    end

    r10 = 2(u + im * (u2 - w2)) * e2iu +
        2emiu * (4u * (2 - im * u) * cosw +
        im * (9u2 + w2 - im * u * (3u2 + w2)) * ξ0)
    r11 = 2(1 + 2im * u) * e2iu + emiu * (-2(1 - im * u) * cosw +
        im * (6u + im * (w2 - 3u2)) * ξ0)
    r12 = 2im * e2iu + im * emiu * (cosw - 3(1 - im * u) * ξ0)
    r20 = -2 * e2iu + 2im * u * emiu * (cosw +
        (1 + 4im * u) * ξ0 + 3u2 * ξ1)
    r21 = -im * emiu * (cosw + (1 + 2im * u) * ξ0 -
        3u2 * ξ1)
    r22 = emiu * (ξ0 - 3im * u * ξ1)

    bdenom = isapprox(0.0, 9u2 - w2) ? 0.0 : 1 / 2(9u2 - w2)^2

    if signflip
        b10 = conj((2u * r10 + (3u2 - w2) * r20 - 2(15u2 + w2) * f0)) * bdenom
        b20 = -conj((r10 - 3u * r20 - 24u * f0)) * bdenom
        b11 = -conj((2u * r11 + (3u2 - w2) * r21 - 2(15u2 + w2) * f1)) * bdenom
        b21 = conj((r11 - 3u * r21 - 24u * f1)) * bdenom
        b12 = conj((2u * r12 + (3u2 - w2) * r22 - 2(15u2 + w2) * f2)) * bdenom
        b22 = -conj((r12 - 3u * r22 - 24u * f2)) * bdenom
    else
        b10 = (2u * r10 + (3u2 - w2) * r20 - 2(15u2 + w2) * f0) * bdenom
        b20 = (r10 - 3u * r20 - 24u * f0) * bdenom
        b11 = (2u * r11 + (3u2 - w2) * r21 - 2(15u2 + w2) * f1) * bdenom
        b21 = (r11 - 3u * r21 - 24u * f1) * bdenom
        b12 = (2u * r12 + (3u2 - w2) * r22 - 2(15u2 + w2) * f2) * bdenom
        b22 = (r12 - 3u * r22 - 24u * f2) * bdenom
    end

    return f0, f1, f2, b10, b11, b12, b20, b21, b22
end

function set_fj(u, w, signflip)
    w2 = w^2
    u2 = u^2
    #if abs(w) <= 0.05
    #    w2 = w2
    #    ξ0 = 1 + 1/6 * w2 * (1 - 1/20 * w2 * (1 - 1/42 * w2))
    if isapprox(0.0, abs(w))
        ξ0 = 1.0
    else
        ξ0 = sin(w) / w
    end

    e2iu = cis(2u)
    emiu = cis(-u)
    cosw = cos(w)
    denom = 9u2 - w2

    if signflip
        if isapprox(0.0, denom)
            fdenom = 1.0
            h0 = one(ComplexF64)
            h1 = zero(ComplexF64)
            h2 = zero(ComplexF64)
        else
            fdenom = 1 / denom
            h0 = conj((u2 - w2) * e2iu + emiu * (8u2 * cosw + 2im * u * (3u2 + w2) * ξ0))
            h1 = -conj(2u * e2iu - emiu * (2u * cosw - im * (3u2 - w2) * ξ0))
            h2 = conj(e2iu - emiu * (cosw + 3im * u * ξ0))
        end
    else
        if isapprox(0.0, denom)
            fdenom = 1.0
            h0 = one(ComplexF64)
            h1 = zero(ComplexF64)
            h2 = zero(ComplexF64)
        else
            fdenom = 1 / denom
            h0 = (u2 - w2) * e2iu + emiu * (8u2 * cosw + 2im * u * (3u2 + w2) * ξ0)
            h1 = 2u * e2iu - emiu * (2u * cosw - im * (3u2 - w2) * ξ0)
            h2 = e2iu - emiu * (cosw + 3im * u * ξ0)
        end
    end

    f0 = h0 * fdenom
    f1 = h1 * fdenom
    f2 = h2 * fdenom
    return f0, f1, f2, ξ0
end

function set_uw(Q::SMatrix{3,3,ComplexF64,9})
    c0_bare = real(det(Q))
    signflip = c0_bare < 0
    c0 = abs(c0_bare)
    c1 = 0.5 * real(multr(Q, Q))
    c13r = sqrt(c1 / 3)
    c0max = 2 * c13r^3
    Θ = isnan(c0 / c0max) ? acos(1.0) : acos(min(1.0, c0 / c0max))

    u = c13r * cos(Θ/3)
    w = sqrt(c1) * sin(Θ/3)
    return u, w, signflip
end

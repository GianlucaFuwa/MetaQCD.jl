"""
Exponential function of iQ ∈ 𝔰𝔲(3), i.e. Projection of Q onto SU(3) \\
From Morningstar & Peardon (2008) arXiv:hep-lat/0311018v1
"""
function exp_iQ(Q::SMatrix{3,3,ComplexF64,9})
    if norm(Q) > 1e-10
        u, w = set_uw(Q)
        f0, f1, f2, _ = set_fj(u, w)
        expiQ = f0 * eye3 + f1 * Q + f2 * Q^2
        return expiQ
    else
        expiQ = eye3
        return expiQ
    end
end

function set_uw(Q::SMatrix{3,3,ComplexF64,9})
    c0 = 1/3 * tr(Q^3)
    c1 = 1/2 * tr(Q^2)
    c13r = sqrt(c1/3)
    c0max = 2 * c13r^3
    Θ = acos(c0 / c0max)

    u = c13r * cos(Θ/3)
    w = sqrt(c1) * sin(Θ/3)
    return u, w
end

function set_fj(u, w)
    if w == 0
        ξ0 = 1
    else
        ξ0 = sin(w) / w
    end

    e2iu = exp(2im*u)
    emiu = exp(-im*u)
    cosw = cos(w)

    h0 = (u^2 - w^2) * e2iu + emiu * (8 * u^2 * cosw + 2im * u * (3u^2 + w^2) * ξ0)
    h1 = 2 * u * e2iu - emiu * (2 * u * cosw - im * (3u^2 - w^2) * ξ0)
    h2 = e2iu - emiu * (cosw + 3im * u * ξ0)
    
    fden = 1 / (9u^2 - w^2)
    f0 = h0 * fden
    f1 = h1 * fden
    f2 = h2 * fden
    return f0, f1, f2, ξ0
end

function calc_coefficients(Q::SMatrix{3,3,ComplexF64,9})
    u, w = set_uw(Q)
    f0, f1, f2, ξ0 = set_fj(u, w)
    e2iu = exp(2im * u)
    emiu = exp(-im * u)
    cosw = cos(w)
    ξ1 = cosw / w^2 - sin(w) / w^3

    r10 = 2 * (u + im * (u^2 - w^2)) * e2iu + 2emiu * 
        (4u * (2 - im * u) * cosw + im * (9u^2 + w^2 - im * u * (3u^2 + w^2)) * ξ0)
    r11 = 2 * (1 + 2im * u) * e2iu + 
        emiu * (-2 * (1 - im * u) * cosw + im * (6u + im * (w^2 - 3u^2)) * ξ0)
    r12 = 2im * e2iu + im * emiu * (cosw - 3 * (1 - im * u) * ξ0)
    r20 = -2 * e2iu + 2im * u * emiu * (cosw + (1 + 4im * u) * ξ0 + 3u^2 * ξ1)
    r21 = -im * emiu * (cosw + (1 +2im * u) * ξ0 - 3u^2 * ξ1)
    r22 = emiu * (ξ0 - 3im * u * ξ1)

    bdenom = 2 * (9u^2 - w^2)^2
    b10 = (2u * r10 + (3u^2 - w^2) * r20 - 2 * (15u^2 + w^2) * f0) / bdenom
    b20 = (r10 - 3u * r20 - 24u * f0) / bdenom
    b11 = (2u * r11 + (3u^2 - w^2) * r21 - 2 * (15u^2 + w^2) * f1) / bdenom
    b21 = (r11 - 3u * r21 - 24u * f1) / bdenom
    b12 = (2u * r12 + (3u^2 - w^2) * r22 - 2 * (15u^2 + w^2) * f2) / bdenom
    b22 = (r12 - 3u * r22 - 24u * f2) / bdenom
    return f0, f1, f2, b10, b11, b12, b20, b21, b22
end
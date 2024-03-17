# Chiral(Weyl)-Basis γ matrices
const γ₁ = @SArray [
    0 0 0 1
    0 0 1 0
    0 -1 0 0
    -1 0 0 0
]

const γ₂ = @SArray [
    0 0 0 -im
    0 0 im 0
    0 im 0 0
    -im 0 0 0
]

const γ₃ = @SArray [
    0 0 1 0
    0 0 0 -1
    -1 0 0 0
    0 1 0 0
]

const γ₄ = @SArray [
    1 0 0 0
    0 1 0 0
    0 0 -1 0
    0 0 0 -1
]

const γ₅ = @SArray [
    -1 0 0 0
    0 -1 0 0
    0 0 1 0
    0 0 0 1
]

const σ₁ = @SArray [
    0 1
    1 0
]

const σ₂ = @SArray [
    0 -im
    im 0
]

const σ₃ = @SArray [
    1 0
    0 -1
]

const λ₁ = @SArray [
    0.0+0.0im 1.0+0.0im 0.0+0.0im
    1.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 0.0+0.0im
]

const λ₂ = @SArray [
    0.0+0.0im 0.0-1.0im 0.0+0.0im
    0.0+1.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 0.0+0.0im
]

const λ₃ = @SArray [
    1.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im -1.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 0.0+0.0im
]

const λ₄ = @SArray [
    0.0+0.0im 0.0+0.0im 1.0+0.0im
    0.0+0.0im 0.0+0.0im 0.0+0.0im
    1.0+0.0im 0.0+0.0im 0.0+0.0im
]

const λ₅ = @SArray [
    0.0+0.0im 0.0+0.0im 0.0-1.0im
    0.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+1.0im 0.0+0.0im 0.0+0.0im
]

const λ₆ = @SArray [
    0.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 1.0+0.0im
    0.0+0.0im 1.0+0.0im 0.0+0.0im
]

const λ₇ = @SArray [
    0.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 0.0-1.0im
    0.0+0.0im 0.0+1.0im 0.0+0.0im
]

const λ₈ = @SArray [
    1/sqrt(3) 0.0+0.0im 0.0+0.0im
    0.0+0.0im 1/sqrt(3) 0.0+0.0im
    0.0+0.0im 0.0+0.0im -2/sqrt(3)
]

const λ = (λ₁, λ₂, λ₃, λ₄, λ₅, λ₆, λ₇, λ₈)

function expλ₁(α)
    φ = 0.5 * α
    sinφ = sin(φ)
    cosφ = cos(φ)

    out = @SMatrix [
        cosφ im*sinφ 0.0+0.0im
        im*sinφ cosφ 0.0+0.0im
        0.0+0.0im 0.0+0.0im 1.0+0.0im
    ]
    return out
end

function expλ₂(α)
    φ = 0.5 * α
    sinφ = sin(φ)
    cosφ = cos(φ)

    out = @SMatrix [
        cosφ sinφ 0.0+0.0im
        -sinφ cosφ 0.0+0.0im
        0.0+0.0im 0.0+0.0im 1.0+0.0im
    ]
    return out
end

function expλ₃(α)
    φ = 0.5 * α
    expiφ = cis(φ)

    out = @SMatrix [
        expiφ 0.0+0.0im 0.0+0.0im
        0.0+0.0im conj(expiφ) 0.0+0.0im
        0.0+0.0im 0.0+0.0im 1.0+0.0im
    ]
    return out
end

function expλ₄(α)
    φ = 0.5 * α
    sinφ = sin(φ)
    cosφ = cos(φ)

    out = @SMatrix [
        cosφ 0.0+0.0im im*sinφ
        0.0+0.0im 1.0+0.0im 0.0+0.0im
        im*sinφ 0.0+0.0im cosφ
    ]
    return out
end

function expλ₅(α)
    φ = 0.5 * α
    sinφ = sin(φ)
    cosφ = cos(φ)

    out = @SMatrix [
        cosφ 0.0+0.0im sinφ
        0.0+0.0im 1.0+0.0im 0.0+0.0im
        -sinφ 0.0+0.0im cosφ
    ]
    return out
end

function expλ₆(α)
    φ = 0.5 * α
    sinφ = sin(φ)
    cosφ = cos(φ)

    out = @SMatrix [
        1.0+0.0im 0.0+0.0im 0.0+0.0im
        0.0+0.0im cosφ im*sinφ
        0.0+0.0im im*sinφ cosφ
    ]
    return out
end

function expλ₇(α)
    φ = 0.5 * α
    sinφ = sin(φ)
    cosφ = cos(φ)

    out = @SMatrix [
        1.0+0.0im 0.0+0.0im 0.0+0.0im
        0.0+0.0im cosφ sinφ
        0.0+0.0im -sinφ cosφ
    ]
    return out
end

function expλ₈(α)
    φ = 0.5 / sqrt(3) * α
    expiφ = cis(φ)

    out = @SMatrix [
        expiφ 0.0+0.0im 0.0+0.0im
        0.0+0.0im expiφ 0.0+0.0im
        0.0+0.0im 0.0+0.0im 1/cis(2φ)
    ]
    return out
end

function expλ(i, α)
    if i == 1
        return expλ₁(α)
    elseif i == 2
        return expλ₂(α)
    elseif i == 3
        return expλ₃(α)
    elseif i == 4
        return expλ₄(α)
    elseif i == 5
        return expλ₅(α)
    elseif i == 6
        return expλ₆(α)
    elseif i == 7
        return expλ₇(α)
    elseif i == 8
        return expλ₈(α)
    else
        return eye3(Float64)
    end
end

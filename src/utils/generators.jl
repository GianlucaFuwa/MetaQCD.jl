# Dirac-Basis γ matrices
const γ1 = @SArray [
    0 0 0 1 
    0 0 1 0
    0 -1 0 0
    -1 0 0 0
]

const γ2 = @SArray [
    0 0 0 -im 
    0 0 im 0
    0 im 0 0
    -im 0 0 0
]

const γ3 = @SArray [
    0 0 1 0 
    0 0 0 -1
    -1 0 0 0
    0 1 0 0
]

const γ4 = @SArray [
    1 0 0 0 
    0 1 0 0
    0 0 -1 0
    0 0 0 -1
]

const σ1 = @SArray [
        0 1
        1 0
]

const σ2 = @SArray [
    0  -im
    im  0
]

const σ3 = @SArray [
    1  0
    0 -1
]

const λ1 = @SArray [
    0.0+0.0im 1.0+0.0im 0.0+0.0im
    1.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 0.0+0.0im
]

const λ2 = @SArray [
    0.0+0.0im 0.0-1.0im 0.0+0.0im
    0.0+1.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 0.0+0.0im
]

const λ3 = @SArray [
    1.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im -1.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 0.0+0.0im
]

const λ4 = @SArray [
    0.0+0.0im 0.0+0.0im 1.0+0.0im
    0.0+0.0im 0.0+0.0im 0.0+0.0im
    1.0+0.0im 0.0+0.0im 0.0+0.0im
]

const λ5 = @SArray [
    0.0+0.0im 0.0+0.0im 0.0-1.0im
    0.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+1.0im 0.0+0.0im 0.0+0.0im
]

const λ6 = @SArray [
    0.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 1.0+0.0im
    0.0+0.0im 1.0+0.0im 0.0+0.0im
]

const λ7 = @SArray [
    0.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 0.0-1.0im
    0.0+0.0im 0.0+1.0im 0.0+0.0im
]

const λ8 = @SArray [
    1/sqrt(3) 0.0+0.0im 0.0+0.0im
    0.0+0.0im 1/sqrt(3) 0.0+0.0im
    0.0+0.0im 0.0+0.0im -2/sqrt(3)
]

function λ(i)
    if i == 1
        return λ1
    elseif i == 2
        return λ2
    elseif i == 3
        return λ3
    elseif i == 4
        return λ4
    elseif i == 5
        return λ5
    elseif i == 6
        return λ6
    elseif i == 7
        return λ7
    elseif i == 8
        return λ8
    else 
        return zero3
    end
end

function expλ1(α)
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

function expλ2(α)
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

function expλ3(α)
    φ = 0.5 * α
    expiφ = cis(φ)

    out = @SMatrix [
        expiφ 0.0+0.0im 0.0+0.0im
        0.0+0.0im conj(expiφ) 0.0+0.0im
        0.0+0.0im 0.0+0.0im 1.0+0.0im
    ]
    return out
end

function expλ4(α)
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

function expλ5(α)
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

function expλ6(α)
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

function expλ7(α)
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

function expλ8(α)
    φ = 0.5/sqrt(3) * α
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
        return expλ1(α)
    elseif i == 2
        return expλ2(α)
    elseif i == 3
        return expλ3(α)
    elseif i == 4
        return expλ4(α)
    elseif i == 5
        return expλ5(α)
    elseif i == 6
        return expλ6(α)
    elseif i == 7
        return expλ7(α)
    elseif i == 8
        return expλ8(α)
    else 
        return eye3
    end
end
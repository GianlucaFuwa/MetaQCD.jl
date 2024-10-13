# INFO: Chiral(Weyl)-Basis γ matrices

@inline function γ1(::Type{T}) where {T}
    return @SArray [
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, -1)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, -1) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 1) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 1) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0)
    ]
end

@inline function γ2(::Type{T}) where {T}
    return @SArray [
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(-1, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(1, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(1, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(-1, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0)
    ]
end

@inline function γ3(::Type{T}) where {T}
    return @SArray [
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, -1) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 1)
        Complex{T}(0, 1) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, -1) Complex{T}(0, 0) Complex{T}(0, 0)
    ]
end

@inline function γ4(::Type{T}) where {T}
    return @SArray [
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(1, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(1, 0)
        Complex{T}(1, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(1, 0) Complex{T}(0, 0) Complex{T}(0, 0)
    ]
end

@inline function γ5(::Type{T}) where {T}
    return @SArray [
        Complex{T}(1, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(1, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(-1, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(-1, 0)
    ]
end

const σ1 = @SArray [
    0 1
    1 0
]

const σ2 = @SArray [
    0 -im
    im 0
]

const σ3 = @SArray [
    1 0
    0 -1
]

# σ_μν = i/2 * [γ_μ, γ_ν]
@inline function σ12(::Type{T}) where {T}
    return @SArray [
        Complex{T}(-1, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(1, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(-1, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(1, 0)
    ]
end

@inline function σ13(::Type{T}) where {T}
    return @SArray [
        Complex{T}(0, 0) Complex{T}(0, -1) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 1) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, -1)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 1) Complex{T}(0, 0)
    ]
end

@inline function σ14(::Type{T}) where {T}
    return @SArray [
        Complex{T}(0, 0) Complex{T}(1, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(1, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(-1, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(-1, 0) Complex{T}(0, 0)
    ]
end

@inline function σ23(::Type{T}) where {T}
    return @SArray [
        Complex{T}(0, 0) Complex{T}(-1, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(-1, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(-1, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(-1, 0) Complex{T}(0, 0)
    ]
end

@inline function σ24(::Type{T}) where {T}
    return @SArray [
        Complex{T}(0, 0) Complex{T}(0, -1) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 1) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 1)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, -1) Complex{T}(0, 0)
    ]
end

@inline function σ34(::Type{T}) where {T}
    return @SArray [
        Complex{T}(1, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(-1, 0) Complex{T}(0, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(-1, 0) Complex{T}(0, 0)
        Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(0, 0) Complex{T}(1, 0)
    ]
end

@generated function σ(::Val{μ}, ::Val{ν}) where {μ,ν}
    return Symbol(:σ, "$μ$ν")
end

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

const λ = (λ1, λ2, λ3, λ4, λ5, λ6, λ7, λ8)

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
        return eye3(Float64)
    end
end

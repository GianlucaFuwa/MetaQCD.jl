using Base.Math: isinf_real
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
    # convenient struct to store exponential coefficients
    struct ExpiQCoeffs{T}
        Q::SU{3,9,T}
        expiQ::SU{3,9,T}
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

    function ExpiQCoeffs(::Type{T}) where {T}
        Q = zero3(T)
        expiQ = zero3(T)
        T0 = zero(ComplexF64)
        return ExpiQCoeffs{T}(Q, expiQ, T0, T0, T0, T0, T0, T0, T0, T0, T0)
    end

    @inline function Base.convert(::Type{ExpiQCoeffs{Tout}}, e::ExpiQCoeffs) where {Tout}
        Q = SU{3,9,Tout}(e.Q)
        expiQ = SU{3,9,Tout}(e.expiQ)
        vals = (e.f₀, e.f₁, e.f₂, e.b₁₀, e.b₁₁, e.b₁₂, e.b₂₀, e.b₂₁, e.b₂₂)
        return ExpiQCoeffs{Tout}(Q, expiQ, vals...)
    end

    @inline function Base.convert(
        ::Type{ExpiQCoeffs{Tout}}, ::Type{ExpiQCoeffs{Tin}}
    ) where {Tin,Tout<:AbstractFloat}
        return ExpiQCoeffs{Tout}
    end

    @inline function Base.convert(::Type{Tout}, e::ExpiQCoeffs) where {Tout<:AbstractFloat}
        return convert(ExpiQCoeffs{Tout}, e)
    end

    @inline function Base.convert(
        ::Type{Tout}, ::Type{ExpiQCoeffs{Tin}}
    ) where {Tin,Tout<:AbstractFloat}
        return ExpiQCoeffs{Tout}
    end

    Base.eltype(::ExpiQCoeffs{T}) where {T} = Complex{T}
    Base.eltype(::Type{ExpiQCoeffs{T}}) where {T} = Complex{T}
    Base.zero(::Type{ExpiQCoeffs{T}}) where {T} = ExpiQCoeffs(T)
    exp_iQ(e::ExpiQCoeffs{T}) where {T} = e.expiQ
    get_Q(e::ExpiQCoeffs{T}) where {T} = e.Q
    get_B₁(e::ExpiQCoeffs{T}) where {T} = Complex{T}(e.b₁₀) * eye3(T) + Complex{T}(e.b₁₁) * e.Q + Complex{T}(e.b₁₂) * cmatmul_oo(e.Q, e.Q)
    get_B₂(e::ExpiQCoeffs{T}) where {T} = Complex{T}(e.b₂₀) * eye3(T) + Complex{T}(e.b₂₁) * e.Q + Complex{T}(e.b₂₂) * cmatmul_oo(e.Q, e.Q)

    """
        exp_iQ(Q::SU{3,9,T}) where {T}
        exp_iQ(e::ExpiQCoeffs{T}) where {T}
    
    Compute the exponential of a traceless Hermitian 3x3 matrix `Q` or return the `exp_iQ` field
    of the `ExpiQCoeffs{T}`-object `e`. \\
    From Morningstar & Peardon (2008) arXiv:hep-lat/0311018v1
    """
    @inline function exp_iQ(Q::SU{3,9,T}) where {T}
        u, w, signflip = set_uw(Q)
        f₀, f₁, f₂, _ = set_fj(u, w, signflip)
        mat = Complex{T}(f₀) * eye3(T) + Complex{T}(f₁) * Q + Complex{T}(f₂) * cmatmul_oo(Q, Q)
        return mat
    end

    """
        exp_iQ_coeffs(Q::SU{3,9,T}) where {T}
    
    Return a `ExpiQCoeffs` object that contains the exponential of `Q` and all parameters
    obtained in the Cayley-Hamilton algorithm that are needed for Stout force recursion.
    """
    @inline function exp_iQ_coeffs(Q::SU{3,9,T}) where {T}
        f₀, f₁, f₂, b₁₀, b₁₁, b₁₂, b₂₀, b₂₁, b₂₂ = calc_coefficients(Q)
        mat = Complex{T}(f₀) * eye3(T) + Complex{T}(f₁) * Q + Complex{T}(f₂) * cmatmul_oo(Q, Q)
        return ExpiQCoeffs(Q, mat, f₀, f₁, f₂, b₁₀, b₁₁, b₁₂, b₂₀, b₂₁, b₂₂)
    end

    function calc_coefficients(Q::SU{3,9,T}) where {T}
        u, w, signflip = set_uw(Q)
        f₀, f₁, f₂, ξ₀ = set_fj(u, w, signflip)
        e²ⁱᵘ = cis(2u)
        e⁻ⁱᵘ = cis(-u)
        cosw = cos(w)
        w² = w * w
        u² = u * u

        if abs(w) <= 0.2
            ξ₁ = -T(1 / 3) + w² / 30 * (one(T) - w² / 28 * (one(T) - w² / 54))
        else
            ξ₁ = cosw / w² - sin(w) / (w² * w)
        end

        r₁₀ =
            2(u + im * (u² - w²)) * e²ⁱᵘ +
            2e⁻ⁱᵘ * (4u * (2 - im * u) * cosw + im * (9u² + w² - im * u * (3u² + w²)) * ξ₀)
        r₁₁ =
            2(1 + 2im * u) * e²ⁱᵘ +
            e⁻ⁱᵘ * (-2(1 - im * u) * cosw + im * (6u + im * (w² - 3u²)) * ξ₀)
        r₁₂ = 2im * e²ⁱᵘ + im * e⁻ⁱᵘ * (cosw - 3(1 - im * u) * ξ₀)
        r₂₀ = -2e²ⁱᵘ + 2im * u * e⁻ⁱᵘ * (cosw + (1 + 4im * u) * ξ₀ + 3u² * ξ₁)
        r₂₁ = -im * e⁻ⁱᵘ * (cosw + (1 + 2im * u) * ξ₀ - 3u² * ξ₁)
        r₂₂ = e⁻ⁱᵘ * (ξ₀ - 3im * u * ξ₁)

        # INFO: bdenom_raw can get very large for some configs (instantons) so we check it
        bdenom_raw = 1 / 2(9u² - w²)^2 
        bdenom = bdenom_raw > 1e6 ? zero(w) : 1 / 2(9u² - w²)^2

        if signflip
            b₁₀ = conj((2u * r₁₀ + (3u² - w²) * r₂₀ - 2(15u² + w²) * conj(f₀))) * bdenom
            b₂₀ = -conj((r₁₀ - 3u * r₂₀ - 24u * conj(f₀))) * bdenom
            b₁₁ = -conj((2u * r₁₁ + (3u² - w²) * r₂₁ - 2(15u² + w²) * -conj(f₁))) * bdenom
            b₂₁ = conj((r₁₁ - 3u * r₂₁ - 24u * -conj(f₁))) * bdenom
            b₁₂ = conj((2u * r₁₂ + (3u² - w²) * r₂₂ - 2(15u² + w²) * conj(f₂))) * bdenom
            b₂₂ = -conj((r₁₂ - 3u * r₂₂ - 24u * conj(f₂))) * bdenom
        else
            b₁₀ = (2u * r₁₀ + (3u² - w²) * r₂₀ - 2(15u² + w²) * f₀) * bdenom
            b₂₀ = (r₁₀ - 3u * r₂₀ - 24u * f₀) * bdenom
            b₁₁ = (2u * r₁₁ + (3u² - w²) * r₂₁ - 2(15u² + w²) * f₁) * bdenom
            b₂₁ = (r₁₁ - 3u * r₂₁ - 24u * f₁) * bdenom
            b₁₂ = (2u * r₁₂ + (3u² - w²) * r₂₂ - 2(15u² + w²) * f₂) * bdenom
            b₂₂ = (r₁₂ - 3u * r₂₂ - 24u * f₂) * bdenom
        end

        return f₀, f₁, f₂, b₁₀, b₁₁, b₁₂, b₂₀, b₂₁, b₂₂
    end

    function set_fj(u, w, signflip)
        w² = w * w
        u² = u * u
        #if abs(w) <= 0.05
        #    w² = w²
        #    ξ₀ = 1 + 1/6 * w² * (1 - 1/20 * w² * (1 - 1/42 * w²))
        # end
        ξ₀ = iszero(w) ? one(w) : (isinf_real(w) ? zero(w) : sin(w) / (w))

        e²ⁱᵘ = cis(2u)
        e⁻ⁱᵘ = cis(-u)
        cosw = cos(w)
        denom = 9u² - w²

        if signflip
            if isinf_real(one(Float64)/denom)
                fdenom = one(Float64)
                h₀ = one(ComplexF64)
                h₁ = zero(ComplexF64)
                h₂ = zero(ComplexF64)
            else
                fdenom = one(Float64) / denom
                h₀ = conj(
                    (u² - w²) * e²ⁱᵘ + e⁻ⁱᵘ * (8u² * cosw + 2im * u * (3u² + w²) * ξ₀)
                )
                h₁ = -conj(2u * e²ⁱᵘ - e⁻ⁱᵘ * (2u * cosw - im * (3u² - w²) * ξ₀))
                h₂ = conj(e²ⁱᵘ - e⁻ⁱᵘ * (cosw + 3im * u * ξ₀))
            end
        else
            if isinf_real(one(Float64)/denom)
                fdenom = one(Float64)
                h₀ = one(ComplexF64)
                h₁ = zero(ComplexF64)
                h₂ = zero(ComplexF64)
            else
                fdenom = one(Float64) / denom
                h₀ = (u² - w²) * e²ⁱᵘ + e⁻ⁱᵘ * (8u² * cosw + 2im * u * (3u² + w²) * ξ₀)
                h₁ = 2u * e²ⁱᵘ - e⁻ⁱᵘ * (2u * cosw - im * (3u² - w²) * ξ₀)
                h₂ = e²ⁱᵘ - e⁻ⁱᵘ * (cosw + 3im * u * ξ₀)
            end
        end

        f₀ = h₀ * fdenom
        f₁ = h₁ * fdenom
        f₂ = h₂ * fdenom
        return f₀, f₁, f₂, ξ₀
    end

    function set_uw(Q::SU{3,9,T}) where {T}
        oneover3 = 1/3
        c₀_bare = real(det(Q))
        signflip = c₀_bare < 0
        c₀ = abs(c₀_bare)
        c₁ = 0.5 * real(multr(Q, Q))
        c₁_3r = sqrt(c₁ * oneover3)
        c₀ᵐᵃˣ = 2(c₁_3r * c₁_3r * c₁_3r)
        Θ = isnan(c₀ / c₀ᵐᵃˣ) ? acos(1) : acos(min(1, c₀ / c₀ᵐᵃˣ))

        u = c₁_3r * cos(Θ * oneover3)
        w = sqrt(c₁) * sin(Θ * oneover3)
        return u, w, signflip
    end
end

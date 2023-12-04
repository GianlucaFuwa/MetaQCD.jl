struct Kernel
    height::Float64
    center::Float64
    σ::Float64
end

(k::Kernel)(s, cutoff², penalty) =
    evaluate_kernel(s, k.height, k.center, k.σ, cutoff², penalty)

@inline function evaluate_kernel(s, height, center, σ, cutoff², penalty)
    diff = (center - s)/σ
    diff² = diff^2
    out = ifelse(diff²>=cutoff², 0.0, height*(exp(-0.5diff²) - penalty))
    return out
end

derivative(k::Kernel, s, cutoff², penalty) =
    kernel_derivative(s, k.height, k.center, k.σ, cutoff², penalty)

@inline function kernel_derivative(s, height, center, σ, cutoff², penalty)
    diff = (center - s)/σ
    diff² = diff^2
    val = ifelse(diff²>=cutoff², 0.0, height*(exp(-0.5diff²) - penalty))
    out = -diff/σ * val
    return out
end

Base.:*(c::Real, k::Kernel) = Kernel(c*k.height, k.center, k.σ)

function merge(k::Kernel, other::Kernel) # Kernel merger
    h = k.height + other.height
    c = (k.height*k.center + other.height*other.center) / h
    s_my_part = k.height * (k.σ^2+k.center^2)
    s_other_part = other.height * (other.σ^2+other.center^2)
    s² = (s_my_part + s_other_part)/h - c^2
    return Kernel(h, c, sqrt(s²))
end

struct Kernel
    height::Float64
    center::Float64
    σ::Float64
end

(k::Kernel)(cv, cutoff², penalty) =
    evaluate_kernel(cv, k.height, k.center, k.σ, cutoff², penalty)

@inline function evaluate_kernel(cv, height, center, σ, cutoff², penalty)
    diff = (center - cv)/σ
    diff² = diff^2
    out = ifelse(diff²>=cutoff², 0.0, height * (exp(-0.5diff²) - penalty))
    return out
end

derivative(k::Kernel, cv, cutoff², penalty) =
    kernel_derivative(cv, k.height, k.center, k.σ, cutoff², penalty)

@inline function kernel_derivative(cv, height, center, σ, cutoff², penalty)
    diff = (center - cv)/σ
    diff² = diff^2
    val = ifelse(diff²>=cutoff², 0.0, height * (exp(-0.5diff²) - penalty))
    out = -diff*val
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

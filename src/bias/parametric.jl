"""
    Parametric(cvlims, penalty_weight, Q, A, Z)
    Parametric(p::ParameterSet; instance=1)

Create an instance of a static Parametric bias using the inputs or the
parameters given in `p`.

# Specifiable parameters
`cvlims::NTuple{2, Float64} = (-6, 6)` - Minimum and maximum of the explorable cv-space;
must be ordered \\
`penalty_weight::Float64 = 1000` - Penalty when cv is outside of `cvlims`; must be positive \\
`Q::Float64 = 0` - Quadratic term in the bias \\
`A::Float64 = 0` - Amplitude of the cosine term in the bias \\
`Z::Float64 = 0` - Frequency of the cosine term in the bias \\
"""
struct Parametric <: AbstractBias
    cvlims::NTuple{2,Float64}
    penalty_weight::Float64
    Q::Float64
    A::Float64
    Z::Float64
end

function Parametric(p::ParameterSet; instance=1)
    cvlims = instance > 0 ? p.cvlims : (-Inf, Inf)
    @level1("|  CVLIMS: $(cvlims)")
    penalty_weight = instance > 0 ? p.penalty_weight : 0.0
    @level1("|  PENALTY WEIGHT: $(penalty_weight)")

    Q, A, Z = instance > 0 ? (p.bias_Q, p.bias_A, p.bias_Z) : (0.0, 0.0, 0.0)
    @level1("|  PARAMETERS: $Q, $A, $Z")
    return Parametric(cvlims, penalty_weight, Q, A, Z)
end

update!(::Parametric, cv, args...) = nothing
clear!(::Parametric) = nothing

function (p::Parametric)(cv)
    Q, A, Z = p.Q, p.A, p.Z
    pen = p.penalty_weight
    lb, ub = p.cvlims
    Q′ = Z * cv

    if in_bounds(cv, lb, ub)
        return Q * cv^2 + A * cos(2π * Q′)
    elseif cv < lb
        penalty = Q * cv^2 + A * cos(2π * Q′) + pen * (cv - lb)^2
        return penalty
    else
        penalty = Q * cv^2 + A * cos(2π * Q′) + pen * (cv - ub)^2
        return penalty
    end
end

function ∂V∂Q(p::Parametric, cv)
    Q, A, Z = p.Q, p.A, p.Z
    pen = p.penalty_weight
    lb, ub = p.cvlims
    Q′ = Z * cv

    if in_bounds(cv, lb, ub)
        return 2Q * cv - 2π * Z * A * sin(2π * Q′)
    elseif cv < lb
        penalty = 2Q * cv - 2π * Z * A * sin(2π * Q′) + 2pen * (cv - lb)
        return penalty
    else
        penalty = 2Q * cv - 2π * Z * A * sin(2π * Q′) + 2pen * (cv - ub)
        return penalty
    end
end

function integral(p::Parametric, lb, ub) # XXX: Why does this function exist?
    Q, A, Z = p.Q, p.A, p.Z
    num = 3A * sin(2π*Z*ub) + 2π*Q*Z*ub^3 - 3A * sin(2π*Z*lb) - 2π*Q*Z*lb^3
    denom = 6π * Z
    return num / denom
end

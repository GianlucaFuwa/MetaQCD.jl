struct Parametric <: AbstractBias
    cvlims::NTuple{2, Float64}
    penalty_weight::Float64
    Q::Float64
    A::Float64
    Z::Float64
end

function Parametric(p::ParameterSet; verbose=Verbose1(), instance=1)
    println_verbose1(verbose, ">> Setting Parametric Bias instance $(instance)...")
    cvlims = instance>0 ? p.cvlims : (-Inf, Inf)
    println_verbose1(verbose, "\t>> CVLIMS = $(cvlims)")
    penalty_weight = instance>0 ? p.penalty_weight : 0.0
    println_verbose1(verbose, "\t>> PENALTY WEIGHT = $(penalty_weight)")

    Q, A, Z = instance>0 ? (p.bias_Q, p.bias_A, p.bias_Z) : (0.0, 0.0, 0.0)
    println_verbose1(verbose, "\t>> PARAMETERS = $Q, $A, $Z")
    return Parametric(cvlims, penalty_weight, Q, A, Z)
end

update!(::Parametric, cv, args...) = nothing
clear!(::Parametric) = nothing
write_to_file(::Parametric, args...) = nothing

function (p::Parametric)(cv)
    Q, A, Z = p.Q, p.A, p.Z
    pen = p.penalty_weight
    lb, ub = p.cvlims
    Q′ = Z * cv

    if in_bounds(cv, lb, ub)
        return    Q*cv^2 + A*cos(2π*Q′)
    elseif cv < lb
        penalty = Q*cv^2 + A*cos(2π*Q′) + pen*(cv - lb)^2
        return penalty
    else
        penalty = Q*cv^2 + A*cos(2π*Q′) + pen*(cv - ub)^2
        return penalty
    end
end

function ∂V∂Q(p::Parametric, cv)
    Q, A, Z = p.Q, p.A, p.Z
    pen = p.penalty_weight
    lb, ub = p.cvlims
    Q′ = Z * cv

    if in_bounds(cv, lb, ub)
        return    2Q*cv - 2π*Z*A*sin(2π*Q′)
    elseif cv < lb
        penalty = 2Q*cv - 2π*Z*A*sin(2π*Q′) + 2pen*(cv - lb)
        return penalty
    else
        penalty = 2Q*cv - 2π*Z*A*sin(2π*Q′) + 2pen*(cv - ub)
        return penalty
    end
end

function (p::Parametric)(cv, t)
    Q, A, Z = p.Q, p.A, p.Z
    pen = p.penalty_weight
    lb, ub = p.cvlims
    Q′ = Z*cv + t

    if in_bounds(Q′, lb, ub)
        return    Q*cv^2 - A*cos(2π*Q′) * sin(π*t)^2
    elseif cv < lb
        penalty = Q*cv^2 - A*cos(2π*Q′) + pen*(cv - lb)^2
        return penalty
    else
        penalty = Q*cv^2 - A*cos(2π*Q′) + pen*(cv - ub)^2
        return penalty
    end
end

function ∂V∂Q(p::Parametric, cv, t)
    Q, A, Z = p.Q, p.A, p.Z
    pen = p.penalty_weight
    lb, ub = p.cvlims
    Q′ = Z*cv + t

    if in_bounds(Q′, lb, ub)
        return    2Q*cv + 2π*Z*A*sin(2π*Q′) * sin(π*t)^2
    elseif cv < lb
        penalty = 2Q*cv + 2π*Z*A*sin(2π*Q′) * sin(π*t)^2 + 2pen*(cv - lb)
        return penalty
    else
        penalty = 2Q*cv + 2π*Z*A*sin(2π*Q′) * sin(π*t)^2 + 2pen*(cv - ub)
        return penalty
    end
end

function ∂V∂t(p::Parametric, cv, t)
    A, Z = p.A, p.Z
    Q′ = Z*cv + t
    return -π*A*(sin(2π*(2t + Z*cv)) - sin(2π*Q′)) #2π*A*sin(2π*Q′)
end

function integral(p::Parametric, lb, ub)
    Q, A, Z = p.Q, p.A, p.Z
    num = 3A*sin(2π*Z*ub) + 2π*Q*Z*ub^3 - 3A*sin(2π*Z*lb) - 2π*Q*Z*lb^3
    denom = 6π * Z
    return num / denom
end

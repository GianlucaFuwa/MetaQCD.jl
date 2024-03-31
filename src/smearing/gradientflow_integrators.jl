struct Euler <: AbstractIntegrator end

function flow!(::Euler, method::GradientFlow)
    Uflow = method.Uflow
    Z = method.Z

    for _ = 1:method.steps
        calcZ!(Z, Uflow, method.ϵ)
        updateU!(Uflow, Z, 1.0)
    end

    return nothing
end

struct RK2 <: AbstractIntegrator end

function flow!(::RK2, method::GradientFlow)
    Uflow = method.Uflow
    Z = method.Z

    for _ = 1:method.steps
        calcZ!(Z, Uflow, method.ϵ)
        updateU!(Uflow, Z, 0.5)
        updateZ!(Z, Uflow, -0.5, method.ϵ)
        updateU!(Uflow, Z, 1.0)
    end

    return nothing
end

struct RK3 <: AbstractIntegrator end

function flow!(::RK3, method::GradientFlow)
    Uflow = method.Uflow
    Z = method.Z

    for _ = 1:method.steps
        calcZ!(Z, Uflow, method.ϵ)
        updateU!(Uflow, Z, 0.25)
        updateZ!(Z, Uflow, -17/36, 8/9 * method.ϵ)
        updateU!(Uflow, Z, 1.0)
        updateZ!(Z, Uflow, -1.0, 3/4 * method.ϵ)
        updateU!(Uflow, Z, 1.0)
    end

    return nothing
end

struct RK3W7 <: AbstractIntegrator end

function flow!(::RK3W7, method::GradientFlow)
    Uflow = method.Uflow
    Z = method.Z

    for _ = 1:method.steps
        calcZ!(Z, Uflow, method.ϵ)
        updateU!(Uflow, Z, 1/3)
        updateZ!(Z, Uflow, -25/48, 15/16 * method.ϵ)
        updateU!(Uflow, Z, 1.0)
        updateZ!(Z, Uflow, -17/25, 8/15 * method.ϵ)
        updateU!(Uflow, Z, 1.0)
    end

    return nothing
end

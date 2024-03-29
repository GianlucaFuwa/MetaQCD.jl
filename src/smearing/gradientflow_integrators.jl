struct Euler <: AbstractIntegrator end

function flow!(::Euler, method::GradientFlow)
    Uflow = method.Uflow
    force = method.Z

    for _ = 1:method.steps
        calc_Z!(Uflow, force, method.ϵ)
        updateU!(Uflow, force, 1.0)
    end

    return nothing
end

struct RK2 <: AbstractIntegrator end

function flow!(::RK2, method::GradientFlow)
    Uflow = method.Uflow
    force = method.Z

    for _ = 1:method.steps
        calc_Z!(Uflow, force, method.ϵ)
        updateU!(Uflow, force, 0.5)
        updateZ!(Uflow, force, -0.5, method.ϵ)
        updateU!(Uflow, force, 1.0)
    end

    return nothing
end

struct RK3 <: AbstractIntegrator end

function flow!(::RK3, method::GradientFlow)
    Uflow = method.Uflow
    force = method.Z

    for _ = 1:method.steps
        calc_Z!(Uflow, force, method.ϵ)
        updateU!(Uflow, force, 0.25)
        updateZ!(Uflow, force, -17/36, 8/9 * method.ϵ)
        updateU!(Uflow, force, 1.0)
        updateZ!(Uflow, force, -1.0, 3/4 * method.ϵ)
        updateU!(Uflow, force, 1.0)
    end

    return nothing
end

struct RK3W7 <: AbstractIntegrator end

function flow!(::RK3W7, method::GradientFlow)
    Uflow = method.Uflow
    force = method.Z

    for _ = 1:method.steps
        calc_Z!(Uflow, force, method.ϵ)
        updateU!(Uflow, force, 1/3)
        updateZ!(Uflow, force, -25/48, 15/16 * method.ϵ)
        updateU!(Uflow, force, 1.0)
        updateZ!(Uflow, force, -17/25, 8/15 * method.ϵ)
        updateU!(Uflow, force, 1.0)
    end

    return nothing
end

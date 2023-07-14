abstract type AbstractIntegrator end

struct GradientFlow{TI, TG} <: AbstractSmearing
    numflow::Int64
    steps::Int64
    ϵ::Float64
    tf::Float64
    measure_every::Int64
    Uflow::TG
    Z::Liefield
end

include("gradientflow_integrators.jl")

function GradientFlow(
    U::TG;
    integrator = "euler",
    numflow = 1,
    steps = 1,
    tf = 0.12,
    measure_every = 1,
) where {TG}
    Z = Liefield(U)
    Uflow = similar(U)

    if integrator == "euler"
        TI = Euler
    elseif integrator == "rk2"
        TI = RK2
    elseif integrator == "rk3"
        TI = RK3
    elseif integrator == "rk3w7"
        TI = RK3W7
    else
        error("Gradient flow integrator \"$(integrator)\" not supported")
    end

    ϵ = tf / steps

    return GradientFlow{TI, TG}(
        numflow,
        steps,
        ϵ,
        tf,
        measure_every,
        Uflow,
        Z,
    )
end

function flow!(method::GradientFlow{TI, TG}) where {TI, TG}
    integrator! = TI()
    integrator!(method)
    return nothing
end


function updateU!(U, Z, ϵ)
    @batch for site in eachindex(U)
        @inbounds for μ in 1:4
            U[μ][site] = cmatmul_oo(
                exp_iQ(-im * ϵ * Z[μ][site]),
                U[μ][site],
            )
        end
    end

    return nothing
end

function calc_Z!(U, Z, ϵ)
    staple = WilsonGaugeAction()

    @batch for site in eachindex(U)
        @inbounds for μ in 1:4
            A = staple(U, μ, site)
            AU = cmatmul_od(A, U[μ][site])
            Z[μ][site] = ϵ * traceless_antihermitian(AU)
        end
    end

    return nothing
end

function updateZ!(U, Z, ϵ_old, ϵ_new)
    staple = WilsonGaugeAction()

    @batch for site in eachindex(U)
        @inbounds for μ in 1:4
            A = staple(U, μ, site)
            AU = cmatmul_od(A, U[μ][site])
            Z[μ][site] = ϵ_old * Z[μ][site] +
                ϵ_new * traceless_antihermitian(AU)
        end
    end

    return nothing
end

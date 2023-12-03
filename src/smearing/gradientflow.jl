abstract type AbstractIntegrator end

struct GradientFlow{TI,TG} <: AbstractSmearing
    numflow::Int64
    steps::Int64
    ϵ::Float64
    tf::Float64
    measure_at::Vector{Int64}
    Uflow::TG
    Z::Liefield
end

include("gradientflow_integrators.jl")

function GradientFlow(U::TG, integrator="euler", numflow=1, steps=1, tf=0.12;
                      measure_every = 1) where {TG}
    @level1("┌ Setting Gradient Flow...")
    Z = Liefield(U)
    Uflow = similar(U)

    integrator = Unicode.normalize(integrator, casefold=true)
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

    if measure_every isa Int64
        measure_at = range(measure_every, numflow, step=measure_every)
    elseif measure_every isa Vector{Int64}
        measure_at = measure_every
    end

    ϵ = tf / steps

    @level1("|  GFLOW INTEGRATOR: $(TI)")
    @level1("|  NUMBER OF GFLOWS: $(numflow)")
    @level1("|  FLOW TIME PER GFLOW: $(tf)")
    @level1("|  INTEGRATION STEPS PER GFLOW: $(steps)")
    @level1("|  INTEGRATION STEP SIZE: $(ϵ)")
    @level1("|  MEASURING ON GFLOW NUMBERS: $(measure_at)")
    @level1("└\n")
    return GradientFlow{TI,TG}(numflow, steps, ϵ, tf, measure_at, Uflow, Z)
end

flow!(method::GradientFlow{TI,TG}) where {TI,TG} = flow!(TI(), method)

function updateU!(U, Z, ϵ)
    @batch per=thread for site in eachindex(U)
        for μ in 1:4
            U[μ][site] = cmatmul_oo(exp_iQ(-im * ϵ * Z[μ][site]), U[μ][site])
        end
    end

    return nothing
end

function calc_Z!(U, Z, ϵ)
    @batch per=thread for site in eachindex(U)
        for μ in 1:4
            A = staple(WilsonGaugeAction(), U, μ, site)
            AU = cmatmul_od(A, U[μ][site])
            Z[μ][site] = ϵ * traceless_antihermitian(AU)
        end
    end

    return nothing
end

function updateZ!(U, Z, ϵ_old, ϵ_new)
    @batch per=thread for site in eachindex(U)
        for μ in 1:4
            A = staple(WilsonGaugeAction(), U, μ, site)
            AU = cmatmul_od(A, U[μ][site])
            Z[μ][site] = ϵ_old * Z[μ][site] + ϵ_new * traceless_antihermitian(AU)
        end
    end

    return nothing
end

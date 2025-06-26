abstract type AbstractIntegrator end

"""
	GradientFlow(U::Gaugefield; numflow=1, steps=1, tf=0.12, measure_every=1)

Create a gradient flow struct that smears `numflow` times with a flow of `tf` and
a step size of `tf / steps`. When used in `calc_measurements_flowed` observables are
after every integer multiple of `measure_every * tf` if `measure_every` is a positive
integer and at `measure_every[i] * tf for i in eachindex(measure_every)` if `measure_every`
is a range.
"""
struct GradientFlow{TI,TG,TT} <: AbstractSmearing
    numflow::Int64
    steps::Int64
    ϵ::Float64
    tf::Float64
    measure_at::Vector{Int64}
    Uflow::TG
    Z::TT
    function GradientFlow(
        U::TG; integrator="euler", numflow=1, steps=1, tf=0.12, measure_every=1
    ) where {TG}
        (numflow == 0 || tf == 0) && (return NoSmearing())

        @level1("- Constructing Gradient Flow...")
        Z = Colorfield(U; no_halo=true)
        Uflow = similar(U)

        integrator = Unicode.normalize(integrator; casefold=true)
        TI = if integrator == "euler"
            Euler
        elseif integrator == "rk2"
            RK2
        elseif integrator == "rk3"
            RK3
        elseif integrator == "rk3w7"
            RK3W7
        else
            error("Gradient flow integrator \"$(integrator)\" not supported")
        end

        measure_at = if measure_every isa Int64
            range(measure_every, numflow; step=measure_every)
        elseif measure_every isa Vector{Int64}
            measure_every
        end

        ϵ = tf / steps

        @level1("|  GFLOW INTEGRATOR: $(string(TI))")
        @level1("|  NUMBER OF GFLOWS: $(numflow)")
        @level1("|  FLOW TIME PER GFLOW: $(tf)")
        @level1("|  INTEGRATION STEPS PER GFLOW: $(steps)")
        @level1("|  INTEGRATION STEP SIZE: $(ϵ)")
        @level1("|  MEASURING ON GFLOW NUMBERS: $(string(measure_at))")
        @level1("-\n")
        return new{TI,TG,typeof(Z)}(numflow, steps, ϵ, tf, measure_at, Uflow, Z)
    end
end

include("gradientflow_integrators.jl")

flow!(gflow::GradientFlow{TI}) where {TI} = flow!(TI(), gflow)

function flow!(gflow::GradientFlow{TI}, Uin) where {TI}
    copy!(gflow.Uflow, Uin)
    flow!(TI(), gflow)
    return nothing
end

function updateU!(U::Gaugefield{B,T}, Z::Colorfield{B,T}, ϵ) where {B,T}
    ϵ = T(ϵ)

    parallelfor(eachindex(U), B) do site
        for μ in 1:4
            U[μ, site] = cmatmul_oo(exp_iQ(-im * ϵ * Z[μ, site]), U[μ, site])
        end
    end

    return nothing
end

function calcZ!(Z::Colorfield{B,T}, U::Gaugefield{B,T}, ϵ) where {B,T}
    ϵ = T(ϵ)
    # TODO: can hide
    update_halo!(U)

    parallelfor(eachindex(U), B) do site
        for μ in 1:4
            A = staple(WilsonGaugeAction(), U, μ, site)
            AU = cmatmul_od(A, U[μ, site])
            Z[μ, site] = ϵ * traceless_antihermitian(AU)
        end
    end

    return nothing
end

function updateZ!(Z::Colorfield{B,T}, U::Gaugefield{B,T}, ϵ_old, ϵ_new) where {B,T}
    ϵ_old = T(ϵ_old)
    ϵ_new = T(ϵ_new)
    # TODO: can hide
    update_halo!(U)

    parallelfor(eachindex(U), B) do site
        for μ in 1:4
            A = staple(WilsonGaugeAction(), U, μ, site)
            AU = cmatmul_od(A, U[μ, site])
            Z[μ, site] = ϵ_old * Z[μ, site] + ϵ_new * traceless_antihermitian(AU)
        end
    end

    return nothing
end

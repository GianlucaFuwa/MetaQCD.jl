abstract type AbstractIntegrator end

struct GradientFlow{TI,TG} <: AbstractSmearing
    numflow::Int64
    steps::Int64
    ϵ::Float64
    measure_every::Int64
    Uflow::Gaugefield{TG}
    Z::Liefield
end

include("gradientflow_integrators.jl")

function GradientFlow(
    U::Gaugefield{TG};
    integrator = "euler",
    numflow = 1,
    steps = 1,
    ϵ = 0.01,
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

    return GradientFlow{TI,TG}(
        numflow,
        steps,
        ϵ,
        measure_every,
        Uflow,
        Z,
    )
end

function flow!(method::GradientFlow{TI,TG}) where {TI,TG}
    integrator! = TI()
    integrator!(method)
    return nothing
end


function updateU!(U, Z, ϵ)
    NX, NY, NZ, NT = size(U)

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    @inbounds for μ in 1:4
                        U[μ][ix,iy,iz,it] = cmatmul_oo(
                            exp_iQ(-im * ϵ * Z[μ][ix,iy,iz,it]),
                            U[μ][ix,iy,iz,it],
                        )
                    end
                end
            end
        end
    end

    return nothing
end

function calc_Z!(
    U::Gaugefield{T},
    Z,
    ϵ,
) where {T}
    NX, NY, NZ, NT = size(U)
    staple = T()

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    @inbounds for μ in 1:4
                        A = staple(U, μ, site)
                        AU = cmatmul_od(A, U[μ][site])
                        Z[μ][site] = ϵ * traceless_antihermitian(AU)
                    end

                end
            end
        end
    end

    return nothing
end

function updateZ!(U::Gaugefield{T}, Z, ϵ_old, ϵ_new) where {T}
    NX, NY, NZ, NT = size(U)
    staple = T()

    @batch for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    @inbounds for μ in 1:4
                        A = staple(U, μ, site)
                        AU = cmatmul_od(A, U[μ][site])
                        Z[μ][site] = ϵ_old * Z[μ][site] +
                            ϵ_new * traceless_antihermitian(AU)
                    end

                end
            end
        end
    end

    return nothing
end

# TODO
struct GradientFlow <: Abstractsmearing
    numflow::Int64
    ϵ::Float64
    _Utmp::Gaugefield

    function GradientFlow(
        U::Gaugefield;
        Nflow = 1,
        ϵ = 0.01
    )
end


module AbstractMeasurementModule
    using Base.Threads
    using LinearAlgebra
    using Polyester
    using Unicode
    using ..Utils
    using ..VerbosePrint

    import ..AbstractSmearingModule: calc_smearedU!, StoutSmearing
    import ..Gaugefields: Gaugefield, clover_rect, clover_square, plaquette, wilsonloop

    include("MeasurementParameters.jl")

    abstract type AbstractMeasurement end

    struct MeasurementOutput{T}
        value::T
        outputstring::String

        MeasurementOutput(value, str) = new{typeof(value)}(value, str)
    end

    function get_value(m::MeasurementOutput)
        return m.value
    end

    function get_string(m::MeasurementOutput)
        return m.outputstring
    end

    function measure(measurement::M, itrj, U) where {M<:AbstractMeasurement}
        error("measure with a type $M is not supported")
    end

    include("measure_gauge_action.jl")
    include("measure_plaquette.jl")
    include("measure_polyakov.jl")
    include("measure_wilson_loop.jl")
    include("measure_topological_charge.jl")

end
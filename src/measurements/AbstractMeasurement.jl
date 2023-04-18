module AbstractMeasurement_module
    using LinearAlgebra
    using Polyester
    using Base.Threads
    using ..Utils

    import ..Verbose_print: Verbose_level,println_verbose2,Verbose_1,Verbose_2,Verbose_3
    import ..Gaugefields: Gaugefield,wilsonloop,Site_coords,plaquette,clover_square,clover_rect,plaquette_tracedsum
    include("MeasurementParameters.jl")

    abstract type AbstractMeasurement end

    struct Measurement_output{T}
        value::T
        outputstring::String

        Measurement_output(value, str) = new{typeof(value)}(value,  str)
    end

    function get_value(m::Measurement_output)
        return m.value
    end

    function get_string(m::Measurement_output)
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


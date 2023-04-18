import ..Gaugefields: 
    GaugeAction_Wilson,
    GaugeAction_Symanzik,
    GaugeAction_Iwasaki,
    GaugeAction_DBW2

mutable struct Gauge_action_measurement <: AbstractMeasurement
    filename::Union{Nothing,String}
    factor::Float64
    verbose_print::Union{Nothing,Verbose_level}
    printvalues::Bool
    GA_methods::Vector{String}

    function Gauge_action_measurement(
        U;
        filename = nothing,
        verbose_level = 2,
        printvalues = false,
        GA_methods = ["Wilson"]
    )
        if printvalues
            if verbose_level == 1
                verbose_print = Verbose_1(filename)
            elseif verbose_level == 2
                verbose_print = Verbose_2(filename)
            elseif verbose_level == 3
                verbose_print = Verbose_3(filename)    
            end
        else
            verbose_print = nothing
        end
        factor = 1 / U.NV / 6 / U.β

        return new(
            filename,
            factor,
            verbose_print,
            printvalues,
            GA_methods
        )
    end
end

function Gauge_action_measurement(
    U::Gaugefield,
    params::Action_parameters,
    filename = "Action_gauge.txt")
    return Gauge_action_measurement(
        U,
        filename = filename,
        verbose_level = params.verbose_level,
        printvalues = params.printvalues,
        GA_methods = params.kinds_of_gauge_action
    )
end

function measure(m::M, U; additional_string = "") where {M<:Gauge_action_measurement}
    measurestring = ""
    nummethod = length(m.GA_methods)
    values = Float64[]
    valuedic = Dict{String,Float64}()
    printstring = " " * additional_string
    for i = 1:nummethod
        methodname = m.GA_methods[i]
        if methodname == "Wilson"
            Sgwils = GaugeAction_Wilson(U) * m.factor
            push!(values, Sgwils)
            valuedic["wilson"] = Sgwils
        elseif methodname == "Symanzik"
            Sgsymanzik = GaugeAction_Symanzik(U) * m.factor
            push!(values, Sgsymanzik)
            valuedic["symanzik"] = Sgsymanzik
        elseif methodname == "Iwasaki"
            Sgiwasaki = GaugeAction_Iwasaki(U) * m.factor
            push!(values, Sgiwasaki)
            valuedic["iwasaki"] = Sgiwasaki
        elseif methodname == "DBW2"
            Sgdbw2 = GaugeAction_DBW2(U) * m.factor
            push!(values, Sgdbw2)
            valuedic["dbw2"] = Sgdbw2
        else 
            error("method $methodname is not supported in gauge action measurement")
        end
    end

    for value in values
        printstring *= "$value "
    end

    for i = 1:nummethod
        methodname = m.GA_methods[i]
        if methodname == "Wilson"
            printstring *= "Sgwilson"
        elseif methodname == "Symanzik"
            printstring *= "Sgsymanzik"
        elseif methodname == "Iwasaki"
            printstring *= "Sgiwasaki"
        elseif methodname == "DBW2"
            printstring *= "Sgdbw2"
        else 
            error("method $methodname is not supported in gauge action measurement")
        end
    end

    if m.printvalues
        measurestring = printstring
        println_verbose2(m.verbose_print, measurestring)
    end

    output = Measurement_output(valuedic, measurestring)
    return output
end
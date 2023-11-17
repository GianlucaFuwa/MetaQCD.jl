struct EnergyDensityMeasurement{T} <: AbstractMeasurement
    ED_dict::Dict{String, Float64}
    fp::T

    function EnergyDensityMeasurement(
        ::Gaugefield;
        filename = "",
        printvalues = false,
        ED_methods = ["clover"],
        flow = false,
    )
        ED_dict = Dict{String, Float64}()
        for method in ED_methods
            ED_dict[method] = 0.0
        end

        if printvalues
            fp = open(filename, "w")
            header = ""
            if flow
                header *= @sprintf("%-9s\t%-7s\t%-9s", "itrj", "iflow", "tflow")
            else
                header *= @sprintf("%-9s", "itrj")
            end

            for methodname in ED_methods
                header *= @sprintf("\t%-22s", "E_$(methodname)")
            end

            println(fp, header)
        else
            fp = nothing
        end

        return new{typeof(fp)}(ED_dict, fp)
    end
end

function EnergyDensityMeasurement(U, params::EnergyDensityParameters, filename, flow=false)
    return EnergyDensityMeasurement(
        U;
        filename = filename,
        printvalues = true,
        ED_methods = params.kinds_of_energy_density,
        flow = flow,
    )
end

function measure(m::EnergyDensityMeasurement{T}, U; additional_string="") where {T}
    measurestring = ""
    printstring = @sprintf("%-9s", additional_string)

    for methodname in keys(m.ED_dict)
        E = energy_density(U, methodname)
        m.ED_dict[methodname] = E
    end

    if T == IOStream
        for value in values(m.ED_dict)
            svalue = @sprintf("%+-22.15E", value)
            printstring *= "\t$(svalue)"
        end

        measurestring = printstring
        println(m.fp, measurestring)
        flush(m.fp)
        measurestring *= " # energy_density"
    end

    output = MeasurementOutput(m.ED_dict, measurestring)
    return output
end

function energy_density(U::Gaugefield, methodname::String)
    if methodname == "plaquette"
        E = energy_density(Plaquette(), U)
    elseif methodname == "clover"
        E = energy_density(Clover(), U)
    elseif methodname == "improved"
        E = energy_density(Improved(), U)
    else
        error("Energy density method '$(methodname)' not supported")
    end

    return E
end

function energy_density(::Plaquette, U::Gaugefield)
    @batch per=thread threadlocal=0.0::Float64 for site in eachindex(U)
        for μ in 1:3
            for ν in μ+1:4
                Cμν = plaquette(U, μ, ν, site)
                Fμν = im * traceless_antihermitian(Cμν)
                threadlocal += real(multr(Fμν, Fμν))
            end
        end
    end

    Eplaq = 1/U.NV * sum(threadlocal)
    return Eplaq
end

function energy_density(::Clover, U::Gaugefield)
    @batch per=thread threadlocal=0.0::Float64 for site in eachindex(U)
        for μ in 1:3
            for ν in μ+1:4
                Cμν = clover_square(U, μ, ν, site, 1)
                Fμν = im/4 * traceless_antihermitian(Cμν)
                threadlocal += real(multr(Fμν, Fμν))
            end
        end
    end

    Eclov = 1/U.NV * sum(threadlocal)
    return Eclov
end

function energy_density(::Improved, U::Gaugefield)
    Eclover = energy_density(Clover(), U)
    Erect = energy_density_rect(U)
    return 5/3*Eclover - 1/12*Erect
end

function energy_density_rect(U::Gaugefield)
    @batch per=thread threadlocal=0.0::Float64 for site in eachindex(U)
        for μ in 1:3
            for ν in μ+1:4
                Cμν = clover_rect(U, μ, ν, site, 1, 2)
                Fμν = im/8 * traceless_antihermitian(Cμν)
                threadlocal += real(multr(Fμν, Fμν))
            end
        end
    end

    Erect = 1/U.NV * sum(threadlocal)
    return Erect
end

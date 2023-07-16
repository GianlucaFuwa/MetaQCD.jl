struct EnergyDensityMeasurement <: AbstractMeasurement
    filename::Union{Nothing, String}
    verbose_print::Union{Nothing, VerboseLevel}
    fp::Union{Nothing, IOStream}
    printvalues::Bool
    ED_methods::Vector{String}

    function EnergyDensityMeasurement(
        U;
        filename = nothing,
        verbose_level = 2,
        printvalues = false,
        ED_methods = ["clover"],
        flow = false,
    )
        if printvalues
            fp = open(filename, "w")
            header = ""
            header *= flow ? "itrj\tiflow\ttflow" : "itrj"

            for methodname in ED_methods
                header *= "\tE$(rpad(methodname, 15, " "))"
            end

            println(fp, header)

            if verbose_level == 1
                verbose_print = Verbose1()
            elseif verbose_level == 2
                verbose_print = Verbose2()
            elseif verbose_level == 3
                verbose_print = Verbose3()
            end
        else
            fp = nothing
            verbose_print = nothing
        end

        return new(
            filename,
            verbose_print,
            fp,
            printvalues,
            ED_methods,
        )
    end
end

function EnergyDensityMeasurement(
        U::T,
        params::EnergyDensityParameters,
        filename = "energy_density.txt",
        flow = false,
    ) where {T <: Gaugefield}
    return EnergyDensityMeasurement(
        U,
        filename = filename,
        verbose_level = params.verbose_level,
        printvalues = params.printvalues,
        ED_methods = params.kinds_of_energy_density,
        flow = flow,
    )
end

function measure(m::EnergyDensityMeasurement, U; additional_string = "")
    measurestring = ""
    values = zeros(Float64, length(m.ED_methods))
    valuedic = Dict{String, AbstractFloat}()
    printstring = "$additional_string"

    for (i, methodname) in enumerate(m.ED_methods)
        E = energy_density(U, methodname)
        values[i] = E
        valuedic[methodname] = E
    end

    if m.printvalues
        for value in values
            svalue = @sprintf("%.15E", value)
            printstring *= "\t$(svalue)"
        end

        measurestring = printstring
        # println_verbose2(m.verbose_print, measurestring)
        println(m.fp, measurestring)
        flush(m.fp)
    end

    output = MeasurementOutput(valuedic, measurestring * " # energy_density")
    return output
end

function energy_density(U::T, methodname::String) where {T <: Gaugefield}
    if methodname == "plaquette"
        E = energy_density_plaq(U)
    elseif methodname == "clover"
        E = energy_density_clover(U)
    elseif methodname == "improved"
        E = energy_density_improved(U)
    else
        error("Energy density method '$(methodname)' not supported")
    end

    return E
end

function energy_density_plaq(U::T) where {T <: Gaugefield}
    spacing = 8
    Eplaq = zeros(Float64, nthreads() * spacing)

    @batch for site in eachindex(U)
        for μ in 1:3
            for ν in μ+1:4
                Cμν = plaquette(U, μ, ν, site)
                Fμν = im * traceless_antihermitian(Cμν)

                Eplaq[threadid() * spacing] += real(multr(Fμν, Fμν))
            end
        end
    end

    Eplaq = 1/2U.NV * sum(Eplaq)
    return Eplaq
end

function energy_density_clover(U::T) where {T <: Gaugefield}
    @batch threadlocal=0.0::Float64 for site in eachindex(U)
        for μ in 1:3
            for ν in μ+1:4
                Cμν = clover_square(U, μ, ν, site, 1)
                Fμν = im/4 * traceless_antihermitian(Cμν)
                threadlocal += real(multr(Fμν, Fμν))
            end
        end
    end

    Eclov = 1/2U.NV * sum(threadlocal)
    return Eclov
end

function energy_density_rect(U::T) where {T <: Gaugefield}
    @batch threadlocal=0.0::Float64 for site in eachindex(U)
        for μ in 1:3
            for ν in μ+1:4
                Cμν = clover_rect(U, μ, ν, site, 1, 2)
                Fμν = im/8 * traceless_antihermitian(Cμν)
                threadlocal += real(multr(Fμν, Fμν))
            end
        end
    end

    Erect = 1/2U.NV * sum(threadlocal)
    return Erect
end

function energy_density_improved(U::T) where {T <: Gaugefield}
    Eclover = energy_density_clover(U)
    Erect = energy_density_rect(U)
    return 5/3 * Eclover - 1/12 * Erect
end

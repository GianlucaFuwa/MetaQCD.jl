mutable struct TopologicalChargeMeasurement <: AbstractMeasurement
    filename::Union{Nothing, String}
    verbose_print::Union{Nothing, VerboseLevel}
    printvalues::Bool
    TC_methods::Vector{String}

    function TopologicalChargeMeasurement(
        U;
        filename = nothing,
        verbose_level = 2,
        printvalues = false,
        TC_methods = ["clover"],
    )
        if printvalues
            if verbose_level == 1
                verbose_print = Verbose1(filename)
            elseif verbose_level == 2
                verbose_print = Verbose2(filename)
            elseif verbose_level == 3
                verbose_print = Verbose3(filename)    
            end
        else
            verbose_print = nothing
        end

        return new(
            filename,
            verbose_print,
            printvalues,
            TC_methods,
        )
    end
end

function TopologicalChargeMeasurement(
        U::Gaugefield,
        params::TopologicalChargeParameters,
        filename = "topological_charge.txt",
    )
    return TopologicalChargeMeasurement(
        U,
        filename = filename,
        verbose_level = params.verbose_level,
        printvalues = params.printvalues,
        TC_methods = params.kinds_of_topological_charge,
    )
end

function measure(m::TopologicalChargeMeasurement, U; additional_string = "")
    measurestring = ""
    values = zeros(Float64, length(m.TC_methods))
    valuedic = Dict{String, AbstractFloat}()
    printstring = additional_string

    for (i, methodname) in enumerate(m.TC_methods)
        Q = top_charge(U, methodname)
        values[i] = Q
        valuedic[methodname] = Q
    end

    if m.printvalues
        for value in values
            printstring *= "$(value) "
        end

        printstring *= "# "

        for methodname in m.TC_methods
            if methodname == "plaquette"
                printstring *= "Qplaq "
            elseif methodname == "clover"
                printstring *= "Qclover "
            elseif methodname == "improved"
                printstring *= "Qimproved"
            end
        end

        measurestring = printstring
        println_verbose2(m.verbose_print, printstring)
    end

    output = MeasurementOutput(valuedic, measurestring)
    return output
end

# Topological charge definitions from: https://arxiv.org/pdf/1708.00696.pdf
function top_charge(U::Gaugefield, methodname::String)
    if methodname == "plaquette"
        Q = top_charge_plaq(U)
    elseif methodname == "clover"
        Q = top_charge_clover(U)
    elseif methodname == "improved"
        Q = top_charge_improved(U)
    else
        error("Topological charge method '$(methodname)' not supported")
    end

    return Q
end

function top_charge_plaq(U::Gaugefield)
    NX, NY, NZ, NT = size(U)
    Qplaq = 0.0

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    C12 = plaquette(U, 1, 2, site)
                    F12 = im * traceless_antihermitian(C12)

                    C13 = plaquette(U, 1, 3, site)
                    F13 = im * traceless_antihermitian(C13)

                    C23 = plaquette(U, 2, 3, site)
                    F23 = im * traceless_antihermitian(C23)

                    C14 = plaquette(U, 1, 4, site)
                    F14 = im * traceless_antihermitian(C14)

                    C24 = plaquette(U, 2, 4, site)
                    F24 = im * traceless_antihermitian(C24)

                    C34 = plaquette(U, 3, 4, site)
                    F34 = im * traceless_antihermitian(C34)

                    Qplaq += 
                        real(tr(F12, F34)) - # minus sign from ε-tensor
                        real(tr(F13, F24)) +
                        real(tr(F14, F23))
                end
            end
        end
    end
    # 1/32 -> 1/4 because of trace symmetry absorbing 8 terms of ε-tensor
    Qplaq = 1/4π^2 * Qplaq
    return Qplaq
end

function top_charge_clover(U::Gaugefield)
    NX, NY, NZ, NT = size(U)
    Qclover = 0.0

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    C12 = clover_square(U, 1, 2, site, 1)
                    F12 = im/4 * traceless_antihermitian(C12)

                    C13 = clover_square(U, 1, 3, site, 1)
                    F13 = im/4 * traceless_antihermitian(C13) 
                    
                    C23 = clover_square(U, 2, 3, site, 1)
                    F23 = im/4 * traceless_antihermitian(C23)

                    C14 = clover_square(U, 1, 4, site, 1)
                    F14 = im/4 * traceless_antihermitian(C14)

                    C24 = clover_square(U, 2, 4, site, 1)
                    F24 = im/4 * traceless_antihermitian(C24)

                    C34 = clover_square(U, 3, 4, site, 1)
                    F34 = im/4 * traceless_antihermitian(C34)

                    Qclover += 
                        real(tr(F12, F34)) - # minus sign from ε-tensor
                        real(tr(F13, F24)) +
                        real(tr(F14, F23))
                end
            end
        end
    end
    # 1/32 -> 1/4 because of trace symmetry absorbing 8 terms of ε-tensor
    Qclover = 1/4π^2 * Qclover
    return Qclover
end

function top_charge_rect(U::Gaugefield)
    NX, NY, NZ, NT = size(U)
    Qrect = 0.0

    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    site = SiteCoords(ix, iy, iz, it)

                    C12 = clover_rect(U, 1, 2, site, 1, 2)
                    F12 = im/8 * traceless_antihermitian(C12)

                    C13 = clover_rect(U, 1, 3, site, 1, 2)
                    F13 = im/8 * traceless_antihermitian(C13)
                    
                    C23 = clover_rect(U, 2, 3, site, 1, 2)
                    F23 = im/8 * traceless_antihermitian(C23)

                    C14 = clover_rect(U, 1, 4, site, 1, 2)
                    F14 = im/8 * traceless_antihermitian(C14)

                    C24 = clover_rect(U, 2, 4, site, 1, 2)
                    F24 = im/8 * traceless_antihermitian(C24)

                    C34 = clover_rect(U, 3, 4, site, 1, 2)
                    F34 = im/8 * traceless_antihermitian(C34)
                     
                    Qrect += 
                        real(tr(F12, F34)) - # minus sign from ε-tensor
                        real(tr(F13, F24)) +
                        real(tr(F14, F23))
                end
            end
        end
    end
    # 2/32 -> 2/4 because of trace symmetry absorbing 8 terms of ε-tensor
    Qrect = 2/4π^2 * Qrect
    return Qrect
end

function top_charge_improved(U::Gaugefield)
    Qclover = top_charge_clover(U)
    Qrect = top_charge_rect(U)
    return 5/3 * Qclover - 1/12 * Qrect
end
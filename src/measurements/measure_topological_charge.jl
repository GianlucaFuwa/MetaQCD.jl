mutable struct Topological_charge_measurement <: AbstractMeasurement
    filename::Union{Nothing,String}
    verbose_print::Union{Nothing,Verbose_level}
    printvalues::Bool
    TC_methods::Vector{String}

    function Topological_charge_measurement(
        U::Gaugefield;
        filename = nothing,
        verbose_level = 2,
        printvalues = false,
        TC_methods = ["Clover"],
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

        return new(
            filename,
            verbose_print,
            printvalues,
            TC_methods,
        )
    end
end

function Topological_charge_measurement(
    U::Gaugefield,
    params::TopCharge_parameters,
    filename = "Topological_charge.txt")
    return Topological_charge_measurement(
        U,
        filename = filename,
        verbose_level = params.verbose_level,
        printvalues = params.printvalues,
        TC_methods = params.kinds_of_topological_charge,
    )
end

function measure(m::M, U; additional_string = "") where {M<:Topological_charge_measurement}
    measurestring = ""
    nummethod = length(m.TC_methods)
    values = Float64[]
    valuedic = Dict{String,Float64}()
    printstring = "" * additional_string
    for i = 1:nummethod
        methodname = m.TC_methods[i]
        if methodname == "Plaquette"
            Qplaq = top_charge_plaq(U)
            push!(values, real(Qplaq))
            valuedic["plaquette"] = real(Qplaq)
        elseif methodname == "Clover"
            Qclover = top_charge_clover(U)
            push!(values, real(Qclover))
            valuedic["clover"] = real(Qclover)
        elseif methodname == "Improved"
            Qimproved = top_charge_improved(U)
            push!(values, real(Qimproved))
            valuedic["improved"] = real(Qimproved)
        else 
            error("method $methodname is not supported in topological charge measurement")
        end
    end
    for value in values
        printstring *= "$(value) "
    end
    printstring *= "# "

    for i = 1:nummethod
        methodname = m.TC_methods[i]
        if methodname == "Plaquette"
            printstring *= "Qplaq "
        elseif methodname == "Clover"
            printstring *= "Qclover "
        elseif methodname == "Improved"
            printstring *= "Qimproved"
        else
            error("method $methodname is not supported in topological charge measurement")
        end
    end

    if m.printvalues
        measurestring = printstring
        println_verbose2(m.verbose_print, printstring)
    end

    output = Measurement_output(valuedic, measurestring)
    return output
end

function top_charge(U::Gaugefield, method::String)
    # Top Charge definitions based on: https://arxiv.org/pdf/1708.00696.pdf
    if method == "Plaquette"
        return top_charge_plaq(U)
    elseif method == "Clover"
        return top_charge_clover(U)
    elseif method == "Improved"
        return top_charge_improved(U)
    else 
        error("Topological Charge method $method not supported")
    end
end

function top_charge_plaq(U::Gaugefield)
    space = 8
    NX, NY, NZ, NT = size(U)
    Qplaq = zeros(Float64, nthreads()*space)

    @batch for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    site = Site_coords(ix,iy,iz,it)

                    C12 = plaquette(U, 1, 2, site)
                    F12 = im * Traceless_antihermitian(C12)

                    C13 = plaquette(U, 1, 3, site)
                    F13 = im * Traceless_antihermitian(C13)

                    C23 = plaquette(U, 2, 3, site)
                    F23 = im * Traceless_antihermitian(C23)

                    C14 = plaquette(U, 1, 4, site)
                    F14 = im * Traceless_antihermitian(C14)

                    C24 = plaquette(U, 2, 4, site)
                    F24 = im * Traceless_antihermitian(C24)

                    C34 = plaquette(U, 3, 4, site)
                    F34 = im * Traceless_antihermitian(C34)

                    Qplaq[threadid()*space] += real(tr( F12*F34 + F13*F24 + F14*F23 ))
                end
            end
        end
    end
    # 1/32 -> 1/4 because of trace symmetry absorbing 8 terms
    return 1/4π^2 * sum(Qplaq)
end

function top_charge_clover(U::Gaugefield)
    space = 8
    NX, NY, NZ, NT = size(U)
    Qclover = zeros(Float64, nthreads()*space)

    @batch for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    site = Site_coords(ix,iy,iz,it)

                    C12 = clover_square(U, 1, 2, site)
                    F12 = im/4 * Traceless_antihermitian(C12)

                    C13 = clover_square(U, 1, 3, site)
                    F13 = im/4 * Traceless_antihermitian(C13) 
                    
                    C23 = clover_square(U, 2, 3, site)
                    F23 = im/4 * Traceless_antihermitian(C23)

                    C14 = clover_square(U, 1, 4, site)
                    F14 = im/4 * Traceless_antihermitian(C14)

                    C24 = clover_square(U, 2, 4, site)
                    F24 = im/4 * Traceless_antihermitian(C24)

                    C34 = clover_square(U, 3, 4, site)
                    F34 = im/4 * Traceless_antihermitian(C34)

                    Qclover[threadid()*space] += real(tr( F12*F34 + F13*F24 + F14*F23 ))
                end
            end
        end
    end
    # 1/32 -> 1/4 because of trace symmetry absorbing 8 terms
    return 1/4π^2 * sum(Qclover)
end

function top_charge_rect(U::Gaugefield)
    space = 8
    NX, NY, NZ, NT = size(U)
    Qrect = zeros(Float64,nthreads()*space)

    @batch for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    site = Site_coords(ix,iy,iz,it)

                    C12 = clover_rect(U, 1, 2, site)
                    F12 = im/8 * Traceless_antihermitian(C12)

                    C13 = clover_rect(U, 1, 3, site)
                    F13 = im/8 * Traceless_antihermitian(C13)
                    
                    C23 = clover_rect(U, 2, 3, site)
                    F23 = im/8 * Traceless_antihermitian(C23)

                    C14 = clover_rect(U, 1, 4, site)
                    F14 = im/8 * Traceless_antihermitian(C14)

                    C24 = clover_rect(U, 2, 4, site)
                    F24 = im/8 * Traceless_antihermitian(C24)

                    C34 = clover_rect(U, 3, 4, site)
                    F34 = im/8 * Traceless_antihermitian(C34)
                     
                    Qrect[threadid()*space] += real(tr( F12*F34 + F13*F24 + F14*F23 ))
                end
            end
        end
    end
    # 2/32 -> 2/4 because of trace symmetry absorbing 8 terms
    return 2/4π^2 * sum(Qrect)
end

function top_charge_improved(U::Gaugefield)
    Qclover = top_charge_clover(U)
    Qrect = top_charge_rect(U)
    return 5/3 * Qclover - 1/12 * Qrect
end
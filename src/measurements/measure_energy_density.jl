struct EnergyDensityMeasurement{T} <: AbstractMeasurement
    ED_dict::Dict{String,Float64} # energy density definition => value
    filename::T
    function EnergyDensityMeasurement(
        U::Gaugefield; filename="", ED_methods=["clover"], flow=false
    )
        ED_dict = Dict{String,Float64}()

        for method in ED_methods
            @level1("|    Method: $(method)")

            if method == "plaquette"
                ED_dict["plaquette"] = 0.0
            elseif method == "clover"
                ED_dict["clover"] = 0.0
            elseif method == "improved"
                if is_distributed(U)
                    @assert U.topology.halo_width >= 2 "improved topological charge requires a halo width of at least 2"
                end

                ED_dict["improved"] = 0.0
            else
                error("Topological charge method $method not supported")
            end
        end

        if !isnothing(filename) && filename != ""
            path = filename * MYEXT
            rpath = StaticString(path)
            header = ""

            if flow
                header *= @sprintf("%-11s%-7s%-9s", "itrj", "iflow", "tflow")
            else
                header *= @sprintf("%-11s", "itrj")
            end

            for methodname in ED_methods
                header *= @sprintf("%-25s", "E_$(methodname)")
            end

            if mpi_amroot()
                open(path, "w") do fp
                    println(fp, header)
                end
            end
        else
            rpath = nothing
        end

        T = typeof(rpath)
        return new{T}(ED_dict, rpath)
    end
end

function EnergyDensityMeasurement(U, params::EnergyDensityParameters, filename, flow=false)
    return EnergyDensityMeasurement(
        U;
        filename=filename,
        ED_methods=params.kinds_of_energy_density,
        flow=flow,
    )
end

function measure(
    m::EnergyDensityMeasurement{T}, U, myinstance, itrj, flow=nothing
) where {T}
    ED_dict = m.ED_dict
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    for method in keys(ED_dict)
        ED_dict[method] = energy_density(U, method)
    end

    if mpi_amroot()
        for method in keys(ED_dict)
            E = ED_dict[method]

            if !isnothing(flow)
                @level1("$itrj\t$E # energydensity_$(method)_flow_$(iflow)")
            else
                @level1("$itrj\t$E # energydensity_$(method)")
            end
        end

        if T !== Nothing
            filename = set_ext!(m.filename, myinstance)
            fp = fopen(filename, "a")
            printf(fp, "%-11i", itrj)

            if !isnothing(flow)
                printf(fp, "%-7i", iflow)
                printf(fp, "%-9.5f", τ)
            end

            for value in values(ED_dict)
                printf(fp, "%+-25.15E", value)
            end

            printf(fp, "\n")
            fclose(fp)
        end
    end

    return ED_dict
end

# Energy density definitions from: https://arxiv.org/pdf/1708.00696.pdf
function energy_density(U, methodname::String)
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

function energy_density(::Plaquette, U::Gaugefield{CPU})
    E = 0.0

    @batch reduction = (+, E) for site in eachindex(U)
        for μ in 1:3
            for ν in (μ+1):4
                Cμν = plaquette(U, μ, ν, site)
                Fμν = im * traceless_antihermitian(Cμν)
                E += real(multr(Fμν, Fμν))
            end
        end
    end

    return distributed_reduce(E / U.NV, +, U)
end

function energy_density(::Clover, U::Gaugefield{CPU,T}) where {T}
    fac = im * T(1/4)
    E = 0.0

    @batch reduction = (+, E) for site in eachindex(U)
        for μ in 1:3
            for ν in (μ+1):4
                Cμν = clover_square(U, μ, ν, site, 1)
                Fμν = fac * traceless_antihermitian(Cμν)
                E += real(multr(Fμν, Fμν))
            end
        end
    end

    return distributed_reduce(E / U.NV, +, U)
end

function energy_density(::Improved, U::Gaugefield{CPU})
    Eclover = energy_density(Clover(), U)
    Erect = energy_density_rect(U)
    return 5 / 3 * Eclover - 1 / 12 * Erect
end

function energy_density_rect(U::Gaugefield{CPU,T}) where {T}
    fac = im * T(1/8)
    E = 0.0

    @batch reduction = (+, E) for site in eachindex(U)
        for μ in 1:3
            for ν in (μ+1):4
                Cμν = clover_rect(U, μ, ν, site, 1, 2)
                Fμν = fac * traceless_antihermitian(Cμν)
                E += real(multr(Fμν, Fμν))
            end
        end
    end

    return distributed_reduce(E / U.NV, +, U)
end

struct EnergyDensityMeasurement{T} <: AbstractMeasurement
    ED_dict::Dict{String,Float64} # energy density definition => value
    filename::T
    function EnergyDensityMeasurement(
        ::Gaugefield; filename="", ED_methods=["clover"], flow=false
    )
        ED_dict = Dict{String,Float64}()
        for method in ED_methods
            @level1("|    Method: $(method)")
            ED_dict[method] = 0.0
        end

        if filename !== nothing && filename != ""
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

            open(path, "w") do fp
                println(fp, header)
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
    m::EnergyDensityMeasurement{Nothing}, U, ::Integer, itrj, flow=nothing,
)
    ED_dict = m.ED_dict
    iflow, _ = isnothing(flow) ? (0, 0.0) : flow

    for method in keys(ED_dict)
        E = energy_density(U, method)
        ED_dict[method] = E

        if MYRANK == 0
            if !isnothing(flow)
                @level1("$itrj\t$E # energydensity_$(method)_flow_$(iflow)")
            else
                @level1("$itrj\t$E # energydensity_$(method)")
            end
        end
    end

    return ED_dict
end

function measure(
    m::EnergyDensityMeasurement{T}, U, myinstance, itrj, flow=nothing
) where {T<:AbstractString}
    ED_dict = m.ED_dict
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    for method in keys(ED_dict)
        ED_dict[method] = energy_density(U, method)
    end

    if MYRANK == 0
        filename = set_ext!(m.filename, myinstance)
        fp = fopen(filename, "a")
        printf(fp, "%-11i", itrj::Int64)

        if !isnothing(flow)
            printf(fp, "%-7i", iflow::Int64)
            printf(fp, "%-9.5f", τ::Float64)
        end

        for value in values(ED_dict)
            printf(fp, "%+-25.15E", value::Float64)
        end

        printf(fp, "\n")
        fclose(fp)
    end

    return ED_dict
end

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
    E = 0.0
    fac = im * T(1/4)

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
    E = 0.0
    fac = im * T(1/8)

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

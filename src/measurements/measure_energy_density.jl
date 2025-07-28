struct EnergyDensityMeasurement{T} <: AbstractMeasurement
    ED_dict::Dict{String,Float64} # energy density definition => value
    filename::T
    function EnergyDensityMeasurement(
        U::Gaugefield; filename="", ED_methods=["clover"], flow=NoSmearing()
    )
        ED_dict = Dict{String,Float64}()

        for method in ED_methods
            @level1("|    type: $(method)")

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
            rpath = StaticString(filename)

            if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
                fp = fopen(filename, "w")
                printf(fp, "%-11s", "itrj")

                if flow == true || flow != NoSmearing()
                    printf(fp, "%-7s", "iflow")
                    printf(fp, "%-9s", "tflow")
                end

                for method in keys(ED_dict)
                    printf(fp, "%-25s", "E_$(method)")
                end

                newline(fp)
                fclose(fp)
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
        ED_methods=params.type,
        flow=flow,
    )
end

function measure(
    m::EnergyDensityMeasurement{T},
    U,
    itrj=0,
    flow=nothing;
    mpi_multi_sim=false,
    fstr="",
) where {T}
    ED_dict = m.ED_dict
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    for method in keys(ED_dict)
        ED_dict[method] = energy_density(U, method)
    end

    if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
        for method in keys(ED_dict)
            E = ED_dict[method]

            if !isnothing(flow)
                @level1("$itrj\t$E # energydensity_$(method)$(fstr)_$(τ)")
            else
                @level1("$itrj\t$E # energydensity_$(method)")
            end
        end

        if T !== Nothing
            filename = if mpi_multi_sim
                set_ext!(m.filename)
            else
                m.filename
            end

            fp = fopen(filename, "a")
            printf(fp, "%-11i", itrj)

            if !isnothing(flow)
                printf(fp, "%-7i", iflow)
                printf(fp, "%-9.5f", τ)
            end

            for method in keys(ED_dict)
                printf(fp, "%+-25.15E", ED_dict[method])
            end

            newline(fp)
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

function energy_density(::Plaquette, U::Gaugefield{B,T,M}) where {B,T,M}
    E = parallelfor_sum(eachindex(U), 0.0, B, Val(M), (U,), (), (U,)) do e, site, (U,)
        for μ in 1:3
            for ν in (μ+1):4
                Cμν = plaquette(U, μ, ν, site)
                Fμν = im * traceless_antihermitian(Cμν)
                e += real(multr(Fμν, Fμν))
            end
        end
        e
    end

    return distributed_reduce(E / length(U), +, U)
end

function energy_density(::Clover, U::Gaugefield{B,T,M}) where {B,T,M}
    fac = im * T(1/4)

    E = parallelfor_sum(eachindex(U), 0.0, B, Val(M), (U,), (), (U,)) do e, site, (U,)
        for μ in 1:3
            for ν in (μ+1):4
                Cμν = clover_1x1(U, μ, ν, site)
                Fμν = fac * traceless_antihermitian(Cμν)
                e += real(multr(Fμν, Fμν))
            end
        end
        e
    end

    return distributed_reduce(E / length(U), +, U)
end

function energy_density(::Improved, U::Gaugefield)
    is_distributed(U) && @assert(U.topology.halo_width>=2)
    Eclover = energy_density(Clover(), U)
    Erect = energy_density_rect(U)
    return 5 / 3 * Eclover - 1 / 12 * Erect
end

function energy_density_rect(U::Gaugefield{B,T,M}) where {B,T,M}
    fac = im * T(1/8)

    E = parallelfor_sum(eachindex(U), 0.0, B, Val(M), (U,), (), (U,)) do e, site, (U,)
        for μ in 1:3
            for ν in (μ+1):4
                Cμν = clover_2x1(U, μ, ν, site) + clover_1x2(U, μ, ν, site)
                Fμν = fac * traceless_antihermitian(Cμν)
                e += real(multr(Fμν, Fμν))
            end
        end
        e
    end

    return distributed_reduce(E / length(U), +, U)
end

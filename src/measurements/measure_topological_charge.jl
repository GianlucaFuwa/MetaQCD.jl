struct TopologicalChargeMeasurement{T} <: AbstractMeasurement
    TC_dict::Dict{String,Float64} # topological charge definition => value
    filename::T
    function TopologicalChargeMeasurement(
        U::Gaugefield; filename="", TC_methods=["clover"], flow=NoSmearing()
    )
        TC_dict = Dict{String,Float64}()

        for method in TC_methods
            @level1("|    type: $(method)")

            if method == "plaquette"
                TC_dict["plaquette"] = 0.0
            elseif method == "clover"
                TC_dict["clover"] = 0.0
            elseif method == "improved"
                if is_distributed(U)
                    @assert U.topology.halo_width >= 2 "improved topological charge requires a halo width of at least 2"
                end

                TC_dict["improved"] = 0.0
            else
                error("Topological charge method $method not supported")
            end
        end

        if !isnothing(filename) && filename != "" && mpi_amroot(mpi_comm_instance())
            rpath = StaticString(filename)

            if !is_distributed(U) || mpi_amroot(mpi_comm_instance())
                fp = fopen(filename, "w")
                printf(fp, ITRJ_STR_FMT, "itrj")

                if flow == true || flow != NoSmearing()
                    printf(fp, IFLOW_STR_FMT, "iflow")
                    printf(fp, TFLOW_STR_FMT, "tflow")
                end

                for method in keys(TC_dict)
                    printf(fp, METHOD_STR_FMT, "Q_$(method)")
                end

                newline(fp)
                fclose(fp)
            end
        else
            rpath = nothing
        end

        T = typeof(rpath)
        return new{T}(TC_dict, rpath)
    end
end

function TopologicalChargeMeasurement(
    U, params::TopologicalChargeParameters, filename, flow=false
)
    return TopologicalChargeMeasurement(
        U;
        filename=filename,
        TC_methods=params.type,
        flow=flow,
    )
end

function measure(
    m::TopologicalChargeMeasurement{T},
    U,
    itrj=0,
    flow=nothing;
    mpi_multi_sim=false,
    fstr="",
) where {T}
    TC_dict = m.TC_dict
    iflow, τ = isnothing(flow) ? (0, 0.0) : flow

    for method in keys(TC_dict)
        TC_dict[method] = top_charge(U, method)
    end

    for method in keys(TC_dict)
        Q = TC_dict[method]

        if !isnothing(flow)
            @level1("$itrj\t$Q # topcharge_$(method)$(fstr)_$(τ)")
        else
            @level1("$itrj\t$Q # topcharge_$(method)")
        end
    end

    if T !== Nothing
        filename = if mpi_multi_sim
            set_ext!(m.filename)
        else
            m.filename
        end

        fp = fopen(filename, "a")
        printf(fp, ITRJ_FMT, itrj)

        if !isnothing(flow)
            printf(fp, IFLOW_FMT, iflow)
            printf(fp, TFLOW_FMT, τ)
        end

        for method in keys(TC_dict)
            printf(fp, METHOD_FMT, TC_dict[method])
        end

        newline(fp)
        fclose(fp)
    end

    return TC_dict
end

# Topological charge definitions from: https://arxiv.org/pdf/1708.00696.pdf
function top_charge(U::Gaugefield, methodname::String)
    if methodname == "plaquette"
        Q = top_charge(Plaquette(), U)
    elseif methodname == "clover"
        Q = top_charge(Clover(), U)
    elseif methodname == "improved"
        Q = top_charge(Improved(), U)
    else
        error("Topological charge method '$(methodname)' not supported")
    end

    return Q
end

function top_charge(::Plaquette, U::Gaugefield{B,T,M}) where {B,T,M}
    itr = eachindex(U)
    Q = parallelfor_sum(itr, 0.0, B, Val(M), (U,), (), (U,); do_edges=Val(true)) do q, site, (U,)
        q += top_charge_density_plaq(U, site)
    end

    return distributed_reduce(Q/4π^2, +, U)
end

function top_charge(::Clover, U::Gaugefield{B,T,M}) where {B,T,M}
    itr = eachindex(U)
    Q = parallelfor_sum(itr, 0.0, B, Val(M), (U,), (), (U,); do_edges=Val(true)) do q, site, (U,)
        q += top_charge_density_clover(U, site, Float64)
    end

    return distributed_reduce(Q/4π^2, +, U)
end

function top_charge(::Improved, U::Gaugefield{B,T,M}) where {B,T,M}
    is_distributed(U) && @assert(U.topology.halo_width>=2)
    c₀ = T(5/3)
    c₁ = T(-2/12)
    itr = eachindex(U)
    Q = parallelfor_sum(itr, 0.0, B, Val(M), (U,), (), (U,); do_edges=Val(true)) do q, site, (U,)
        q += top_charge_density_imp(U, site, c₀, c₁, T)
    end

    return distributed_reduce(Q/4π^2, +, U)
end

@inline function top_charge_density_plaq(U, site)
    C₁₂ = plaquette(U, 1, 2, site)
    F₁₂ = C₁₂ - C₁₂'
    C₁₃ = plaquette(U, 1, 3, site)
    F₁₃ = C₁₃ - C₁₃'
    C₂₃ = plaquette(U, 2, 3, site)
    F₂₃ = C₂₃ - C₂₃'
    C₁₄ = plaquette(U, 1, 4, site)
    F₁₄ = C₁₄ - C₁₄'
    C₂₄ = plaquette(U, 2, 4, site)
    F₂₄ = C₂₄ - C₂₄'
    C₃₄ = plaquette(U, 3, 4, site)
    F₃₄ = C₃₄ - C₃₄'

    qₙ = real(multr(F₁₂, F₃₄)) - real(multr(F₁₃, F₂₄)) + real(multr(F₁₄, F₂₃))
    return -qₙ
end

@inline function top_charge_density_clover(U, site, ::Type{T}) where {T}
    C₁₂ = clover_1x1(U, 1, 2, site)
    F₁₂ = C₁₂ - C₁₂'
    C₁₃ = clover_1x1(U, 1, 3, site)
    F₁₃ = C₁₃ - C₁₃'
    C₂₃ = clover_1x1(U, 2, 3, site)
    F₂₃ = C₂₃ - C₂₃'
    C₁₄ = clover_1x1(U, 1, 4, site)
    F₁₄ = C₁₄ - C₁₄'
    C₂₄ = clover_1x1(U, 2, 4, site)
    F₂₄ = C₂₄ - C₂₄'
    C₃₄ = clover_1x1(U, 3, 4, site)
    F₃₄ = C₃₄ - C₃₄'

    qₙ = real(multr(F₁₂, F₃₄)) - real(multr(F₁₃, F₂₄)) + real(multr(F₁₄, F₂₃))
    return -T(1/64) * qₙ
end

@inline function top_charge_density_imp(U, site, c₀, c₁, ::Type{T}) where {T}
    q_clov = top_charge_density_clover(U, site, T)
    q_rect = top_charge_density_rect(U, site, T)
    q_imp = c₀ * q_clov + c₁ * q_rect
    return q_imp
end

@inline function top_charge_density_rect(U, site, ::Type{T}) where {T}
    C₁₂ = clover_2x1(U, 1, 2, site) + clover_1x2(U, 1, 2, site)
    F₁₂ = C₁₂ - C₁₂'
    C₁₃ = clover_2x1(U, 1, 3, site) + clover_1x2(U, 1, 3, site)
    F₁₃ = C₁₃ - C₁₃'
    C₂₃ = clover_2x1(U, 2, 3, site) + clover_1x2(U, 2, 3, site)
    F₂₃ = C₂₃ - C₂₃'
    C₁₄ = clover_2x1(U, 1, 4, site) + clover_1x2(U, 1, 4, site)
    F₁₄ = C₁₄ - C₁₄'
    C₂₄ = clover_2x1(U, 2, 4, site) + clover_1x2(U, 2, 4, site)
    F₂₄ = C₂₄ - C₂₄'
    C₃₄ = clover_2x1(U, 3, 4, site) + clover_1x2(U, 3, 4, site)
    F₃₄ = C₃₄ - C₃₄'

    qₙ = real(multr(F₁₂, F₃₄)) - real(multr(F₁₃, F₂₄)) + real(multr(F₁₄, F₂₃))
    return -T(1/256) * qₙ
end

function top_charge_deriv!(
    dU::Colorfield{B,T}, F::Tensorfield{B,TF,M}, U::Gaugefield{B,TU}, kind_of_charge, fac=1.0
) where {B,T,M,TF,TU}
    c = T(fac / 4π^2)
    fieldstrength_eachsite!(kind_of_charge, F, U) # halo update of U done here

    parallelfor(eachindex(dU, F, U), B, Val(M), (F,), (U,), (F, U); do_edges=Val(true)) do site, (F, U)
        # @inbounds begin
            tmp1 = cmatmul_oo(
                U[1, site],
                (
                    ∇trFμνFρσ(kind_of_charge, U, F, 1, 2, 3, 4, site) -
                    ∇trFμνFρσ(kind_of_charge, U, F, 1, 3, 2, 4, site) +
                    ∇trFμνFρσ(kind_of_charge, U, F, 1, 4, 2, 3, site)
                ),
            )
            dU[1, site] = c * traceless_antihermitian(tmp1)
            tmp2 = cmatmul_oo(
                U[2, site],
                (
                    ∇trFμνFρσ(kind_of_charge, U, F, 2, 3, 1, 4, site) -
                    ∇trFμνFρσ(kind_of_charge, U, F, 2, 1, 3, 4, site) -
                    ∇trFμνFρσ(kind_of_charge, U, F, 2, 4, 1, 3, site)
                ),
            )
            dU[2, site] = c * traceless_antihermitian(tmp2)
            tmp3 = cmatmul_oo(
                U[3, site],
                (
                    ∇trFμνFρσ(kind_of_charge, U, F, 3, 1, 2, 4, site) -
                    ∇trFμνFρσ(kind_of_charge, U, F, 3, 2, 1, 4, site) +
                    ∇trFμνFρσ(kind_of_charge, U, F, 3, 4, 1, 2, site)
                ),
            )
            dU[3, site] = c * traceless_antihermitian(tmp3)
            tmp4 = cmatmul_oo(
                U[4, site],
                (
                    ∇trFμνFρσ(kind_of_charge, U, F, 4, 2, 1, 3, site) -
                    ∇trFμνFρσ(kind_of_charge, U, F, 4, 1, 2, 3, site) -
                    ∇trFμνFρσ(kind_of_charge, U, F, 4, 3, 1, 2, site)
                ),
            )
            dU[4, site] = c * traceless_antihermitian(tmp4)
        # end
    end

    return nothing
end

# """
# Derivative of the FμνFρσ term for Field strength tensor given by plaquette
# """
function ∇trFμνFρσ(::Plaquette, U, F, μ, ν, ρ, σ, site)
    Nμ = axes(U, μ)
    Nν = axes(U, ν)
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteν⁺ = move(site, ν, 1, Nν)
    siteν⁻ = move(site, ν, -1, Nν)
    siteμ⁺ν⁻ = move(siteμ⁺, ν, -1, Nν)
    i = get_tensor_index(ρ, σ)
    sgn = ρ > σ ? -1 : 1

    component =
        cmatmul_oddo(U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site], F[i, site]) +
        cmatmul_ddoo(U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], F[i, siteν⁻], U[ν, siteν⁻])

    return eltype(component)(im * sgn / 2) * component
end

# """
# Derivative of the FμνFρσ term for Field strength tensor given by 1x1-Clover
# """
function ∇trFμνFρσ(::Clover, U, F, μ, ν, ρ, σ, site)
    Nμ = axes(U, μ)
    Nν = axes(U, ν)
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteν⁺ = move(site, ν, 1, Nν)
    siteν⁻ = move(site, ν, -1, Nν)
    siteμ⁺ν⁺ = move(siteμ⁺, ν, 1, Nν)
    siteμ⁺ν⁻ = move(siteμ⁺, ν, -1, Nν)
    i = get_tensor_index(ρ, σ)
    sgn = ρ > σ ? -1 : 1

    # get reused matrices up to cache (can precalculate some products too)
    # Uνsiteμ⁺ = U[ν,siteμ⁺]
    # Uμsiteν⁺ = U[μ,siteν⁺]
    # Uνsite = U[ν,site]
    # Uνsiteμ⁺ν⁻ = U[ν,siteμ⁺ν⁻]
    # Uμsiteν⁻ = U[μ,siteν⁻]
    # Uνsiteν⁻ = U[ν,siteν⁻]

    component =
        cmatmul_oddo(U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site], F[i, site]) +
        cmatmul_odod(U[ν, siteμ⁺], U[μ, siteν⁺], F[i, siteν⁺], U[ν, site]) +
        cmatmul_oodd(U[ν, siteμ⁺], F[i, siteμ⁺ν⁺], U[μ, siteν⁺], U[ν, site]) +
        cmatmul_oodd(F[i, siteμ⁺], U[ν, siteμ⁺], U[μ, siteν⁺], U[ν, site]) -
        cmatmul_ddoo(U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻], F[i, site]) -
        cmatmul_ddoo(U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], F[i, siteν⁻], U[ν, siteν⁻]) -
        cmatmul_dodo(U[ν, siteμ⁺ν⁻], F[i, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻]) -
        cmatmul_oddo(F[i, siteμ⁺], U[ν, siteμ⁺ν⁻], U[μ, siteν⁻], U[ν, siteν⁻])

    return eltype(component)(im * sgn / 8) * component
end

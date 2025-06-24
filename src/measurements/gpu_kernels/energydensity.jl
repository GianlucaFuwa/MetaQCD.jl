function energy_density(::Plaquette, U::Gaugefield{B}) where {B<:GPU}
    bulk = eachindex(U)
    E = @latsum(bulk, Float64, energy_density_plaq_kernel!, U)
    return E / length(U)
end

function energy_density(::Clover, U::Gaugefield{B,T}) where {B<:GPU,T}
    bulk = eachindex(U)
    E = @latsum(bulk, Float64, energy_density_clov_kernel!, U, T)
    return E / length(U)
end

function energy_density(::Improved, U::Gaugefield{B,T}) where {B<:GPU,T}
    bulk = eachindex(U)
    E =  @latsum(bulk, Float64, energy_density_imp_kernel!, U, T)
    return E / length(U)
end

@kernel function energy_density_plaq_kernel!(out, @Const(U))
    # workgroup index, that we use to pass the reduced value to global "out"
    iblock = @index(Group, Linear)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    e = 0.0
    @inbounds for μ in (1i32):(4i32)
        for ν in (μ + 1i32):(4i32)
            if μ == ν
                continue
            end
            Cμν = plaquette(U, μ, ν, site)
            Fμν = im * traceless_antihermitian(Cμν)
            e += real(multr(Fμν, Fμν))
        end
    end

    out_group = @groupreduce(+, e, 0.0)

    ithread = @index(Local)
    if ithread == 1
        @inbounds out[iblock] = out_group
    end
end

@kernel function energy_density_clov_kernel!(out, @Const(U), ::Type{T}) where {T}
    # workgroup index, that we use to pass the reduced value to global "out"
    iblock = @index(Group, Linear)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    fac = im * T(1/4)

    e = 0.0
    @inbounds for μ in (1i32):(4i32)
        for ν in (μ + 1i32):(4i32)
            if μ == ν
                continue
            end
            Cμν = clover_square(U, μ, ν, site, 1)
            Fμν = fac * traceless_antihermitian(Cμν)
            e += real(multr(Fμν, Fμν))
        end
    end

    out_group = @groupreduce(+, e, 0.0)

    ithread = @index(Local)
    if ithread == 1
        @inbounds out[iblock] = out_group
    end
end

@kernel function energy_density_imp_kernel!(out, @Const(U), ::Type{T}) where {T}
    # workgroup index, that we use to pass the reduced value to global "out"
    iblock = @index(Group, Linear)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    fac1 = im * T(1/4)
    fac2 = im * T(1/8)

    ec = 0.0
    er = 0.0
    @inbounds for μ in (1i32):(4i32)
        for ν in (μ + 1i32):(4i32)
            if μ == ν
                continue
            end
            Cμν = clover_square(U, μ, ν, site, 1)
            Fμν = fac1 * traceless_antihermitian(Cμν)
            ec += real(multr(Fμν, Fμν))
            Cμν = clover_rect(U, μ, ν, site, 1, 2)
            Fμν = fac2 * traceless_antihermitian(Cμν)
            er += real(multr(Fμν, Fμν))
        end
    end

    out_group = @groupreduce(+, 5/3 * ec - 1/12 * er, 0.0)

    ithread = @index(Local)
    if ithread == 1
        @inbounds out[iblock] = out_group
    end
end

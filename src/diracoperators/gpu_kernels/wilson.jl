function LinearAlgebra.mul!(
    ψ::TF, D::WilsonDiracOperator{B,T,C,TF,TG,BC}, ϕ::TF
) where {B<:GPU,T,C,TF,TG,BC}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass_term = T(8 + 2 * D.mass)
    csw = D.csw
    bc = D.boundary_condition
    bulk = eachindex(ψ, ϕ, U)
    @latmap(Sequential(), Val(1), wilson_gpu!, ψ, U, ϕ, mass_term, bc, T, Val(1), bulk)

    if has_clover_term(D)
        fac = T(-csw / 2)
        @latmap(Sequential(), Val(1), add_clover_gpu!, ψ, U, ϕ, fac, T, bulk)
    end
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{WilsonDiracOperator{B,T,C,TF,TG,BC}}, ϕ::TF
) where {B<:GPU,T,C,TF,TG,BC}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass_term = T(8 + 2 * D.parent.mass)
    csw = D.parent.csw
    bc = D.parent.boundary_condition
    bulk = eachindex(ψ, ϕ, U)
    @latmap(Sequential(), Val(1), wilson_gpu!, ψ, U, ϕ, mass_term, bc, T, Val(-1), bulk)

    if has_clover_term(D)
        fac = T(-csw / 2)
        @latmap(Sequential(), Val(1), add_clover_gpu!, ψ, U, ϕ, fac, T, bulk)
    end
end

@kernel cpu=false function wilson_gpu!(
    ψ, @Const(U), @Const(ϕ), mass_term, bc, ::Type{T}, ::Val{dagg}
) where {T,dagg}
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    @inbounds ψ[site] = wilson_kernel(U, ϕ, site, mass_term, bc, T, Val(dagg))
end

@kernel cpu=false function add_clover_gpu!(
    ψ, @Const(U), @Const(ϕ), fac, ::Type{T}, bulk
) where {T}
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    @inbounds ψ[site] += clover_kernel(U, ϕ, site, fac, T)
end

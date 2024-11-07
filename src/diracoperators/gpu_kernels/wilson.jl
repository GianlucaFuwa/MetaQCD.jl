function LinearAlgebra.mul!(
    ψ::TF, D::WilsonDiracOperator{B,T,C,TF,TG}, ϕ::TF
) where {B<:GPU,T,C,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    check_dims(ψ, ϕ, U)
    mass_term = T(8 + 2 * D.mass)
    csw = D.csw
    bc = D.boundary_condition
    @latmap(Sequential(), Val(1), wilson_kernel!, ψ, U, ϕ, mass_term, bc, T, Val(1))

    if has_clover_term(D)
        fac = T(-csw / 2)
        @latmap(Sequential(), Val(1), add_clover_kernel!, ψ, U, ϕ, fac, T)
    end
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{WilsonDiracOperator{B,T,C,TF,TG}}, ϕ::TF
) where {B<:GPU,T,C,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    check_dims(ψ, ϕ, U)
    mass_term = T(8 + 2 * D.parent.mass)
    csw = D.parent.csw
    bc = D.parent.boundary_condition
    @latmap(Sequential(), Val(1), wilson_kernel!, ψ, U, ϕ, mass_term, bc, T, Val(-1))

    if has_clover_term(D)
        fac = T(-csw / 2)
        @latmap(Sequential(), Val(1), add_clover_kernel!, ψ, U, ϕ, fac, T)
    end
end

@kernel function wilson_kernel!(
    ψ, @Const(U), @Const(ϕ), mass_term, bc, ::Type{T}, ::Val{dagg}
) where {T,dagg}
    site = @index(Global, Cartesian)
    @inbounds ψ[site] = wilson_kernel(U, ϕ, site, mass_term, bc, T, Val(dagg))
end

@kernel function add_clover_kernel!(ψ, @Const(U), @Const(ϕ), fac, ::Type{T}) where {T}
    site = @index(Global, Cartesian)
    @inbounds ψ[site] += clover_kernel(U, ϕ, site, fac, T)
end

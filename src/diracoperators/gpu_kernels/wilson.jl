function LinearAlgebra.mul!(
    ψ::TF, D::WilsonDiracOperator{B,T,C,TF,TG}, ϕ::TF
) where {B<:GPU,T,C,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass_term = T(8 + 2 * D.mass)
    csw = D.csw
    anti = D.anti_periodic
    check_dims(ψ, ϕ, U)
    @latmap(Sequential(), Val(1), wilson_kernel, ψ, U, ϕ, mass_term, anti, T, Val(1))
    if C
        fac = T(-csw / 2)
        @latmap(Sequential(), Val(1), clover_kernel, ψ, U, ϕ, fac, T)
    end
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{WilsonDiracOperator{B,T,C,TF,TG}}, ϕ::TF
) where {B<:GPU,T,C,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass_term = T(8 + 2 * D.parent.mass)
    csw = D.parent.csw
    anti = D.parent.anti_periodic
    check_dims(ψ, ϕ, U)
    @latmap(Sequential(), Val(1), wilson_kernel, ψ, U, ϕ, mass_term, anti, T, Val(-1))
    if C
        fac = T(-csw / 2)
        @latmap(Sequential(), Val(1), clover_kernel, ψ, U, ϕ, fac, T)
    end
end

@kernel function wilson_kernel(
    ψ, @Const(U), @Const(ϕ), mass_term, anti, ::Type{T}, ::Val{dagg}
) where {T,dagg}
    site = @index(Global, Cartesian)
    @inbounds ψ[site] = wilson_kernel(U, ϕ, site, mass_term, anti, T, Val(dagg))
end

@kernel function clover_kernel(ψ, @Const(U), @Const(ϕ), fac, ::Type{T}) where {T}
    site = @index(Global, Cartesian)
    @inbounds ψ[site] = clover_kernel(U, ϕ, site, fac, T)
end

function LinearAlgebra.mul!(
    ψ::TF, D::StaggeredDiracOperator{B,T,TF,TG}, ϕ::TF
) where {B<:GPU,T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass = T(D.mass)
    anti = D.anti_periodic
    check_dims(ψ, ϕ, U)
    @latmap(Sequential(), Val(1), staggered_kernel, ψ, U, ϕ, mass, anti, T, false)
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{StaggeredDiracOperator{B,T,TF,TG}}, ϕ::TF
) where {B<:GPU,T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass = T(D.parent.mass)
    anti = D.parent.anti_periodic
    check_dims(ψ, ϕ, U)
    @latmap(Sequential(), Val(1), staggered_kernel, ψ, U, ϕ, mass, anti, T, true)
end

@kernel function staggered_kernel(
    ψ, @Const(U), @Const(ϕ), mass, anti, ::Type{T}, dagg
) where {T}
    site = @index(Global, Cartesian)
    @inbounds ψ[site] = staggered_kernel(U, ϕ, site, mass, anti, T, dagg)
end

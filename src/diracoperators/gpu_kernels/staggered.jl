function LinearAlgebra.mul!(
    ψ::TF, D::StaggeredDiracOperator{B,T,TF,TG}, ϕ::TF
) where {B<:GPU,T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass = T(D.mass)
    bc = D.boundary_condition
    check_dims(ψ, ϕ, U)
    @latmap(Sequential(), Val(1), staggered_kernel!, ψ, U, ϕ, mass, bc, T, false)
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{StaggeredDiracOperator{B,T,TF,TG}}, ϕ::TF
) where {B<:GPU,T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    check_dims(ψ, ϕ, U)
    mass = T(D.parent.mass)
    bc = D.parent.boundary_condition
    @latmap(Sequential(), Val(1), staggered_kernel!, ψ, U, ϕ, mass, bc, T, true)
end

@kernel function staggered_kernel!(
    ψ, @Const(U), @Const(ϕ), mass, bc, ::Type{T}, dagg
) where {T}
    site = @index(Global, Cartesian)
    @inbounds ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, T, dagg)
end

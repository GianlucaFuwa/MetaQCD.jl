function LinearAlgebra.mul!(
    ψ::TF, D::StaggeredDiracOperator{B,T,TF,TG,BC}, ϕ::TF
) where {B<:GPU,T,TF,TG,BC}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass = T(D.mass)
    bc = D.boundary_condition
    bulk = eachindex(ψ, ϕ, U)
    # TODO: can hide
    update_halo!(U, ϕ)
    @latmap(bulk, staggered_gpu!, ψ, U, ϕ, mass, bc, T, false)
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{StaggeredDiracOperator{B,T,TF,TG,BC}}, ϕ::TF
) where {B<:GPU,T,TF,TG,BC}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass = T(D.parent.mass)
    bc = D.parent.boundary_condition
    bulk = eachindex(ψ, ϕ, U)
    # TODO: can hide
    update_halo!(U, ϕ)
    @latmap(bulk, staggered_gpu!, ψ, U, ϕ, mass, bc, T, true)
end

@kernel cpu=false function staggered_gpu!(
    ψ, @Const(U), @Const(ϕ), mass, bc, ::Type{T}, dagg, bulk
) where {T}
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    @inbounds ψ[site] = staggered_kernel(U, ϕ, site, mass, bc, T, dagg)
end

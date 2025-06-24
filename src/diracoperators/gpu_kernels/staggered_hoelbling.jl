function LinearAlgebra.mul!(
    ψ::TF, D::StaggeredHoelblingDiracOperator{MT,B,T,TF,TG}, ϕ::TF
) where {MT,B<:GPU,T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.U
    mass = T(D.mass)
    term = get_mass_term(D)
    bc = D.boundary_condition
    bulk = eachindex(ψ, ϕ, U)
    # TODO: can hide
    update_halo!(U, ϕ)
    @latmap(bulk, staggered_hoelbling_gpu!, ψ, U, ϕ, mass, bc, term, T, false)
end

function LinearAlgebra.mul!(
    ψ::TF, D::Daggered{StaggeredDiracOperator{MT,B,T,TF,TG}}, ϕ::TF
) where {MT,B<:GPU,T,TF,TG}
    @assert TG !== Nothing "Dirac operator has no gauge background, do `D(U)`"
    U = D.parent.U
    mass = T(D.parent.mass)
    term = get_mass_term(D.parent)
    bc = D.parent.boundary_condition
    bulk = eachindex(ψ, ϕ, U)
    # TODO: can hide
    update_halo!(U, ϕ)
    @latmap(bulk, staggered_hoelbling_gpu!, ψ, U, ϕ, mass, bc, term, T, true)
end

@kernel cpu=false function staggered_hoelbling_gpu!(
    ψ, @Const(U), @Const(ϕ), mass, bc, term, ::Type{T}, dagg, bulk
) where {T}
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    @inbounds ψ[site] = staggered_hoelbling_kernel(U, ϕ, site, mass, bc, term, T, dagg)
end


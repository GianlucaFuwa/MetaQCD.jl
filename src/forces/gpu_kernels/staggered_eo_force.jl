function add_staggered_eo_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T}, X_eo::TF, Y_eo::TF, bc; coeff=1
) where {B,T,TF}
    X = X_eo.parent
    Y = Y_eo.parent
    fac = T(-0.5coeff)
    bulk = eachindex(dU, U, X, Y)
    # TODO: can hide
    update_halo!(U, X, Y)
    @latmap(bulk, add_staggered_eo_derivative_gpu!, dU, U, X, Y, bc, fac)
    return nothing
end

@kernel cpu=false function add_staggered_eo_derivative_gpu!(
    dU, @Const(U), @Const(X), @Const(Y), bc, fac, bulk
)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    add_staggered_eo_derivative_kernel!(dU, U, X, Y, site, bc, fac, bulk)
end


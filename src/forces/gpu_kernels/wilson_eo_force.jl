function add_wilson_eo_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T}, X_eo::TF, Y_eo::TF, bc; coeff=1
) where {B,T,TF<:WilsonEOPreSpinorfield{CPU,T}}
    fac = T(0.5coeff)
    X = X_eo.parent
    Y = Y_eo.parent
    bulk = eachindex(dU, U, X, Y)
    # TODO: can hide
    update_halo!(U, X, Y)
    @latmap(bulk, add_wilson_eo_derivative_gpu!, dU, U, X, Y, bc, fac)
    return nothing
end

@kernel cpu=false function add_wilson_eo_derivative_gpu!(
    dU, @Const(U), @Const(X), @Const(Y), bc, fac, bulk
)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    add_wilson_eo_derivative_kernel!(dU, U, X, Y, site, bc, fac, bulk)
end

function calc_Xμν_eo_eachsite!(Xμν::Tensorfield{B,T}, X_eo::TF, Y_eo::TF) where {B,T,TF}
    X = X_eo.parent
    Y = Y_eo.parent
    bulk = eachindex(Xμν, X, Y)
    @latmap(bulk, calc_Xμν_eo_eachsite_gpu!, Xμν, X, Y)
    return nothing
end

@kernel cpu=false function calc_Xμν_eo_eachsite_gpu!(Xμν, @Const(X), @Const(Y), bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    calc_Xμν_eo_kernel!(Xμν, X, Y, site, bulk)
end

function calc_small_Xμν_eachsite!(
    Xμν::Tensorfield{B,T}, D_oo_inv::Paulifield{B,T,M,true}
) where {B,T,M}
    bulk = eachindex(Xμν, D_oo_inv)
    @latmap(bulk, calc_small_Xμν_eachsite_gpu!, Xμν, D_oo_inv, T)
    return nothing
end

@kernel cpu=false function calc_small_Xμν_eachsite_gpu!(
    Xμν, @Const(D_oo_inv), ::Type{T}, bulk
) where {T}
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    calc_small_Xμν_kernel!(Xμν, D_oo_inv, site, T, bulk)
end

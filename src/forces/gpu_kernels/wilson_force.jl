function add_wilson_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T}, X::TF, Y::TF, bc; coeff=1
) where {B<:GPU,T,TF<:WilsonSpinorfield{B,T}}
    fac = T(0.5coeff)
    bulk = eachindex(dU, U, X, Y)
    # TODO: can hide
    update_halo!(U, X, Y)
    @latmap(bulk, add_wilson_derivative_gpu!, dU, U, X, Y, bc, fac)
end

@kernel cpu=false function add_wilson_derivative_gpu!(
    dU, @Const(U), @Const(X), @Const(Y), bc, fac, bulk
)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @inbounds begin
        add_wilson_derivative_kernel!(dU, U, X, Y, site, bc, fac)
    end
end

function add_clover_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T}, Xμν::Tensorfield{B,T}, csw; coeff=1
) where {B<:GPU,T}
    fac = T(csw * coeff / 2)
    bulk = eachindex(dU, U, Xμν)
    # TODO: can hide
    update_halo!(Xμν)
    @latmap(bulk, add_clover_derivative_gpu!, dU, U, Xμν, fac, T)
end

@kernel cpu=false function add_clover_derivative_gpu!(
    dU, @Const(U), @Const(Xμν), fac, ::Type{T}, bulk
) where {T}
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @inbounds begin
        add_clover_derivative_kernel!(dU, U, Xμν, site, fac, T)
    end
end

function calc_Xμν_wilson_eachsite!(
    Xμν::Tensorfield{B,T}, X::TF, Y::TF
) where {B<:GPU,T,TF<:WilsonSpinorfield{B,T}}
    bulk = eachindex(Xμν, X, Y)
    @latmap(bulk, calc_Xμν_wilson_gpu!, Xμν, X, Y)
end

@kernel cpu=false function calc_Xμν_wilson_gpu!(Xμν, @Const(X), @Const(Y), bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    calc_Xμν_wilson_kernel!(Xμν, X, Y, site)
end

function add_staggered_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T}, X::TF, Y::TF, bc; coeff=1
) where {B<:GPU,T,TF<:StaggeredSpinorfield{B,T}}
    fac = T(-0.5coeff)
    bulk = eachindex(dU, U, X, Y)
    @latmap(
        Sequential(), Val(1), add_staggered_derivative_gpu!, dU, U, X, Y, bc, fac, bulk
    )
end

@kernel cpu=false function add_staggered_derivative_gpu!(
    dU, @Const(U), @Const(X), @Const(Y), bc, fac, bulk
)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @inbounds begin
        add_staggered_derivative_kernel!(dU, U, X, Y, site, bc, fac)
    end
end

function add_staggered_hoelbling_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T}, X::TF, Y::TF, bc, term; coeff=1
) where {B<:GPU,T,TF<:StaggeredSpinorfield{B,T}}
    check_dims(dU, U, X, Y)
    fac1 = T(-0.5coeff)
    fac2 = T(coeff)
    @latmap(
        Sequential(), Val(1), add_staggered_h_derivative_gpu_kernel!, dU, U, X, Y, bc, term,
        fac1, fac2
    )
end

@kernel function add_staggered_h_derivative_gpu_kernel!(
    dU, @Const(U), @Const(X), @Const(Y), bc, term, fac1, fac2
)
    site = @index(Global, Cartesian)
    _μ, _ν, _ρ, _σ = term

    @inbounds begin
        add_staggered_derivative_kernel!(dU, U, X, Y, site, bc, fac1)
        add_hoelbling_derivative_kernel!(dU, _μ, _ν, U, X, Y, site, bc, fac2)
        add_hoelbling_derivative_kernel!(dU, _ρ, _σ, U, X, Y, site, bc, fac2)
    end
end


function add_staggered_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T}, X::TF, Y::TF, anti; coeff=1
) where {B<:GPU,T,TF<:StaggeredFermionfield{B,T}}
    check_dims(dU, U, X, Y)
    fac = T(-0.5coeff)
    @latmap(Sequential(), Val(1), add_staggered_derivative_gpu_kernel!, dU, U, X, Y, anti, fac)
end

@kernel function add_staggered_derivative_gpu_kernel!(
    dU, @Const(U), @Const(X), @Const(Y), anti, fac
)
    NT = dims(U)[4]
    site = @index(Global, Cartesian)
    bc⁺ = boundary_factor(anti, site[4], 1, NT)

    @inbounds begin
        add_staggered_derivative_kernel!(dU, U, X, Y, site, bc⁺, fac)
    end
end

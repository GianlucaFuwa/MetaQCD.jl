function add_wilson_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T}, X::TF, Y::TF, anti; coeff=1
) where {B<:GPU,T,TF<:WilsonSpinorfield{B,T}}
    check_dims(dU, U, X, Y)
    fac = T(0.5coeff)
    @latmap(Sequential(), Val(1), add_wilson_derivative_gpu_kernel!, dU, U, X, Y, anti, fac)
end

@kernel function add_wilson_derivative_gpu_kernel!(
    dU, @Const(U), @Const(X), @Const(Y), anti, fac
)
    NT = dims(U)[4]
    site = @index(Global, Cartesian)
    bc⁺ = boundary_factor(anti, site[4], 1, NT)

    @inbounds begin
        add_wilson_derivative_kernel!(dU, U, X, Y, site, bc⁺, fac)
    end
end

function add_clover_derivative!(
    dU::Colorfield{B,T}, U::Gaugefield{B,T}, Xμν::Tensorfield{B,T}, csw; coeff=1
) where {B<:GPU,T}
    check_dims(dU, U, Xμν)
    fac = T(csw * coeff / 2)
    @latmap(Sequential(), Val(1), add_clover_derivative_kernel!, dU, U, Xμν, fac, T)
end

@kernel function add_clover_derivative_kernel!(
    dU, @Const(U), @Const(Xμν), fac, ::Type{T}
) where {T}
    site = @index(Global, Cartesian)

    @inbounds begin
        add_clover_derivative_kernel!(dU, U, Xμν, site, fac, T)
    end
end

function calc_Xμν_eachsite!(
    Xμν::Tensorfield{B,T}, X::TF, Y::TF
) where {B<:GPU,T,TF<:WilsonSpinorfield{B,T}}
    check_dims(Xμν, X, Y)
    @latmap(Sequential(), Val(1), calc_Xμν_kernel!, dU, U, Xμν, fac, T)
end

@kernel function calc_Xμν_kernel!(Xμν, @Const(X), @Const(Y))
    site = @index(Global, Cartesian)

    @inbounds begin
        calc_Xμν_kernel!(Xμν, X, Y, site)
    end
end

# FIXME:
function mul_oe!(ψ::TF, U::Gaugefield{B,T}, ϕ::TF, anti, into_odd, dagg) where {B<:GPU,T,TF}
    check_dims(ψ, ϕ, U)
    fdims = dims(ψ)
    NV = ψ.NV
    @latmap(
        Checkerboard2(),
        Val(1),
        staggered_eo_kernel!,
        ψ,
        U,
        ϕ,
        anti,
        into_odd,
        false,
        dagg,
        T,
        fdims,
        NV
    )
end

function mul_eo!(ψ::TF, U::Gaugefield{B,T}, ϕ::TF, anti, into_odd, dagg) where {B<:GPU,T,TF}
    check_dims(ψ, ϕ, U)
    fdims = dims(ψ)
    NV = ψ.NV
    @latmap(
        Checkerboard2(),
        Val(1),
        staggered_eo_kernel!,
        ψ,
        U,
        ϕ,
        anti,
        into_odd,
        true,
        dagg,
        T,
        fdims,
        NV
    )
end

@kernel function staggered_eo_kernel!(
    ψ, @Const(U), @Const(ϕ), anti, into_odd, from_odd, dagg, ::Type{T}, fdims, NV
) where {T}
    iy, iz, it = @index(Global, NTuple)

    for ix in 1i32+iseven(iy + iz + it + from_odd):2i32:size(U, 2i32)
        site = SiteCoords(ix, iy, iz, it)
        _site = if into_odd
            eo_site_switch(site, fdims..., NV)
        else
            eo_site(site, fdims..., NV)
        end
        @inbounds ψ[_site] = staggered_eo_kernel(U, ϕ, site, anti, T, dagg)
    end
end

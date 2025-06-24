function fieldstrength_eachsite!(
    ::Plaquette, F::Tensorfield{B,T}, U::Gaugefield{B,T}
) where {B<:GPU,T}
    update_halo!(U)
    @latmap(eachindex(F, U), fieldstrength_eachsite_plaq_gpu!, F, U)
    return nothing
end

@kernel cpu=false function fieldstrength_eachsite_plaq_gpu!(F, @Const(U))
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @inbounds begin
        C12 = plaquette(U, 1i32, 2i32, site)
        F[1i32, 2i32, site] = im * traceless_antihermitian(C12)
        C13 = plaquette(U, 1i32, 3i32, site)
        F[1i32, 3i32, site] = im * traceless_antihermitian(C13)
        C14 = plaquette(U, 1i32, 4i32, site)
        F[1i32, 4i32, site] = im * traceless_antihermitian(C14)
        C23 = plaquette(U, 2i32, 3i32, site)
        F[2i32, 3i32, site] = im * traceless_antihermitian(C23)
        C24 = plaquette(U, 2i32, 4i32, site)
        F[2i32, 4i32, site] = im * traceless_antihermitian(C24)
        C34 = plaquette(U, 3i32, 4i32, site)
        F[3i32, 4i32, site] = im * traceless_antihermitian(C34)
    end
end

function fieldstrength_eachsite!(
    ::Clover, F::Tensorfield{B,T}, U::Gaugefield{B,T}
) where {B<:GPU,T}
    update_halo!(U)
    @latmap(eachindex(F, U), fieldstrength_eachsite_clover_gpu!, F, U, T)
    return nothing
end

@kernel cpu=false function fieldstrength_eachsite_clover_gpu!(
    F, @Const(U), ::Type{T}, bulk
) where {T}
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    fac = Complex{T}(im / 4)

    @inbounds begin
        C12 = clover_square(U, 1i32, 2i32, site, 1i32)
        F[1i32, 2i32, site] = fac * traceless_antihermitian(C12)
        C13 = clover_square(U, 1i32, 3i32, site, 1i32)
        F[1i32, 3i32, site] = fac * traceless_antihermitian(C13)
        C14 = clover_square(U, 1i32, 4i32, site, 1i32)
        F[1i32, 4i32, site] = fac * traceless_antihermitian(C14)
        C23 = clover_square(U, 2i32, 3i32, site, 1i32)
        F[2i32, 3i32, site] = fac * traceless_antihermitian(C23)
        C24 = clover_square(U, 2i32, 4i32, site, 1i32)
        F[2i32, 4i32, site] = fac * traceless_antihermitian(C24)
        C34 = clover_square(U, 3i32, 4i32, site, 1i32)
        F[3i32, 4i32, site] = fac * traceless_antihermitian(C34)
    end
end

function swap_U!(a::Gaugefield{B}, b::Gaugefield{B}) where {B<:GPU}
    @latmap(eachindex(a, b), swap_U_kernel!, a, b)
    return nothing
end

@kernel cpu=false function swap_U_kernel!(a, b, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    
    @unroll for μ in 1i32:4i32
        @inbounds a_tmp = a[μ, site]
        @inbounds a[μ, site] = b[μ, site]
        @inbounds b[μ, site] = a_tmp
    end
end

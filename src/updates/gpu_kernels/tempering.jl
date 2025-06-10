function swap_U!(a::Gaugefield{B}, b::Gaugefield{B}) where {B<:GPU}
    @latmap(Sequential(), Val(1), swap_U_kernel!, a, b, eachindex(a, b))
    return nothing
end

@kernel function swap_U_kernel!(a, b, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    
    @unroll for μ in 1i32:4i32
        @inbounds a_tmp = a[μ, site]
        @inbounds a[μ, site] = b[μ, site]
        @inbounds b[μ, site] = a_tmp
    end
end

function for_eachindex(f, u::AbstractField{B}, args...) where {B}
    itr = eachindex(u, args...)

    if B == CPU
        @batch for i in eachindex(IndexLinear(), itr)
            @inbounds site = itr[i]
            @inline f(site)
        end
    else
    end
end

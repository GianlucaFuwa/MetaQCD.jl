"""
	@latmap(itr::AbstractIterator, C, kernel, U, GA, fac)
Apply update algorithm `kernel` on each element in `U` following the pattern specified by
`itr` `C` times.
"""
macro latmap(itr, C, f!, U, GA, fac)
    quote
        $__latmap($(esc(itr)), $(esc(C)), $(esc(f!)), $(esc(U)), $(esc(GA)), $(esc(fac)))
    end
end

function __latmap(
    ::Sequential, ::Val{C}, f!::F, U::Gaugefield{B,T,M}, GA, fac
) where {C,F,B,T,M}
    C == 0 && return nothing

    for _ in 1:C
        parallelfor(eachindex(U), B, Val(M), (U,), (U,), (U,)) do site, (U,)
            for μ in 1:4
                f!(U, μ, site, GA, fac)
            end
        end
    end

    return nothing
end

function __latmap(
    ::Checkerboard2, ::Val{C}, f!::F, U::Gaugefield{B,T,M}, GA, fac
) where {C,F,B,T,M}
    C == 0 && return nothing
    NX = get_local_dims(U)[1]
    _, yrange, zrange, trange = U.topology.bulk_sites.indices
    itr = CartesianIndices((yrange, zrange, trange))

    for _ in 1:C
        for μ in 1:4
            for pass in 1:2
                parallelfor(itr, B, Val(M), (U,), (U,), (U,)) do yzt, (U,)
                    for ix in (1 + iseven(sum(yzt.I) + pass)):2:NX
                        site = CartesianIndex((ix, yzt.I...))
                        f!(U, μ, site, GA, fac)
                    end
                end
            end
        end
    end

    return nothing
end

function __latmap(
    ::Checkerboard4, ::Val{C}, f!::F, U::Gaugefield{B,T,M}, GA, fac
) where {C,F,B,T,M}
    C == 0 && return nothing

    for _ in 1:C
        for μ in 1:4
            for pass in 1:4
                parallelfor(eachindex(U), B, Val(M), (U,), (U,), (U,)) do site, (U,)
                    if mod1(sum(site.I) + site[μ], 4) == pass
                        f!(U, μ, site, GA, fac)
                    end
                end
            end
        end
    end

    return nothing
end

"""
	@latsum(itr::AbstractIterator, kernel, U, GA, fac)
Sum update algorithm `kernel` on each element in `U` following the pattern specified by
`itr` `C` times.
"""
macro latsum(itr, C, f!, U, GA, fac)
    quote
        $__latsum($(esc(itr)), $(esc(C)), $(esc(f!)), $(esc(U)), $(esc(GA)), $(esc(fac)))
    end
end

function __latsum(
    ::Sequential, ::Val{C}, f!::F, U::Gaugefield{B,T,M}, GA, fac
) where {C,F,B,T,M}
    C == 0 && return 0.0
    itr = eachindex(U)
    out = 0.0

    for _ in 1:C
        out += parallelfor_sum(itr, 0.0, B, Val(M), (U,), (U,), (U,)) do outi, site, (U,)
            for μ in 1:4
                outi += f!(U, μ, site, GA, fac)
            end
            outi
        end
    end

    return distributed_reduce(out, +, U)
end

function __latsum(
    ::Checkerboard2, ::Val{C}, f!::F, U::Gaugefield{B,T,M}, GA, fac
) where {C,F,B,T,M}
    C == 0 && return 0.0
    NX = get_local_dims(U)[1]
    _, yrange, zrange, trange = U.topology.bulk_sites.indices
    itr = CartesianIndices((yrange, zrange, trange))
    out = 0.0

    for _ in 1:C
        for μ in 1:4
            for pass in 1:2
                out += parallelfor_sum(itr, 0.0, B, Val(M), (U,), (U,), (U,)) do outi, yzt, (U,)
                    for ix in (1 + iseven(sum(yzt.I) + pass)):2:NX
                        site = CartesianIndex((ix, yzt.I...))
                        outi += f!(U, μ, site, GA, fac)
                    end
                    outi
                end
            end
        end
    end

    return distributed_reduce(out, +, U)
end

function __latsum(
    ::Checkerboard4, ::Val{C}, f!::F, U::Gaugefield{B,T,M}, GA, fac
) where {C,F,B,T,M}
    C == 0 && return 0.0
    itr = eachindex(U)
    out = 0.0

    for _ in 1:C
        for μ in 1:4
            for pass in 1:4
                out += parallelfor_sum(itr, 0.0, B, Val(M), (U,), (U,), (U,)) do outi, site, (U,)
                    if mod1(sum(site.I) + site[μ], 4) == pass
                        outi += f!(U, μ, site, GA, fac)
                    end
                    outi
                end
            end
        end
    end

    return distributed_reduce(out, +, U)
end

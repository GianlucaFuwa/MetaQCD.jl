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

function __latmap(::Sequential, ::Val{C}, f!::F, U::AbstractField{CPU}, GA, fac) where {F,C}
    C == 0 && return nothing

    for _ in 1:C
        @batch for site in eachindex(U)
            for μ in 1:4
                f!(U, μ, site, GA, fac)
            end
        end
    end

    return nothing
end

function __latmap(
    ::Checkerboard2, ::Val{C}, f!::F, U::AbstractField{CPU}, GA, fac
) where {F,C}
    C == 0 && return nothing
    NX, NY, NZ, NT = dims(U)

    for _ in 1:C
        for μ in 1:4
            for pass in 1:2
                @batch for ss in CartesianIndices((NY, NZ, NT))
                    for ix in (1 + iseven(sum(ss.I) + pass)):2:NX
                        site = CartesianIndex((ix, ss.I...))
                        f!(U, μ, site, GA, fac)
                    end
                end
            end
        end
    end

    return nothing
end

function __latmap(
    ::Checkerboard4, ::Val{C}, f!::F, U::AbstractField{CPU}, GA, fac
) where {F,C}
    C == 0 && return nothing

    for _ in 1:C
        for μ in 1:4
            for pass in 1:4
                @batch for site in eachindex(U)
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

function __latsum(::Sequential, ::Val{C}, f!::F, U::AbstractField{CPU}, GA, fac) where {C,F}
    C == 0 && return 0.0
    out = 0.0

    for _ in 1:C
        @batch reduction = (+, out) for site in eachindex(U)
            for μ in 1:4
                out += f!(U, μ, site, GA, fac)
            end
        end
    end

    return MPI.Allreduce(out, +, U.comm_cart)
end

function __latsum(
    ::Checkerboard2, ::Val{C}, f!::F, U::AbstractField{CPU}, GA, fac
) where {C,F}
    C == 0 && return 0.0
    NX, NY, NZ, NT = dims(U)
    out = 0.0

    for _ in 1:C
        for μ in 1:4
            for pass in 1:2
                @batch reduction = (+, out) for ss in CartesianIndices((NY, NZ, NT))
                    for ix in (1 + iseven(sum(ss.I) + pass)):2:NX
                        site = CartesianIndex((ix, ss.I...))
                        out += f!(U, μ, site, GA, fac)
                    end
                end
            end
        end
    end

    return MPI.Allreduce(out, +, U.comm_cart)
end

function __latsum(
    ::Checkerboard4, ::Val{C}, f!::F, U::AbstractField{CPU}, GA, fac
) where {C,F}
    C == 0 && return 0.0
    out = 0.0

    for _ in 1:C
        for μ in 1:4
            for pass in 1:4
                @batch reduction = (+, out) for site in eachindex(U)
                    if mod1(sum(site.I) + site[μ], 4) == pass
                        out += f!(U, μ, site, GA, fac)
                    end
                end
            end
        end
    end

    return MPI.Allreduce(out, +, U.comm_cart)
end

macro latmap(itr, f!, U, args...)
    quote
        $__latmap($(esc(itr)), $(esc(f!)), $(esc(U)), $(map(esc, args)...))
    end
end

function __latmap(itr::CartesianIndices, f!::F, U::AbstractField{B}, args...) where {F,B<:GPU}
    ndrange = size(itr)
    workgroupsize = ntuple(i -> min(ndrange[i], 4), Val(4)) # 4^4 = 256 threads per workgroup should be fine
    kernel! = f!(B(), workgroupsize, ndrange)
    kernel!(get_parents(U, args...)..., itr; ndrange=ndrange)
    KA.synchronize(B())
    return nothing
end

macro latsum(itr, f!, U, args...)
    quote
        $__latsum($(esc(itr)), $(esc(f!)), $(esc(U)), $(map(esc, args)...))
    end
end

function __latsum(
    itr::CartesianIndices, ::Type{OutType}, f!::F, U::AbstractField{B}, args...
) where {OutType,F,B<:GPU}
    ndrange = size(itr)
    workgroupsize = ntuple(i -> min(ndrange[i], 4), Val(4))
    numblocks = cld(length(itr), prod(workgroupsize))
    out = KA.zeros(B(), OutType, numblocks)
    # @show out, workgroupsize, numblocks
    kernel! = f!(B(), workgroupsize, ndrange)
    kernel!(out, get_parents(U, args...)..., itr; ndrange=ndrange)
    KA.synchronize(B())
    return sum(out)
end

macro latsup(itr, C, f!, U, args...)
    quote
        $__latsup($(esc(itr)), $(esc(C)), $(esc(f!)), $(esc(U)), $(map(esc, args)...))
    end
end

function __latsup(
    itr, ::Type{OutType}, f!::F, U::AbstractField{B}, args...
) where {OutType,F,B<:GPU}
    ndrange = size(itr)
    workgroupsize = ntuple(i -> min(ndrange[i], 4), Val(4))
    numblocks = cld(length(itr), prod(workgroupsize))
    out = KA.zeros(B(), OutType, numblocks)
    kernel! = f!(B(), workgroupsize, ndrange)
    kernel!(out, get_parents(U, args...)..., itr; ndrange=ndrange)
    KA.synchronize(B())
    return maximum(out)
end

@inline function get_parents(args...)
    return ntuple(i -> args[i] isa SpinorfieldEO ? args[i].parent : args[i], length(args))
end

@inline function get_raws(args...)
    fields = filter(x -> x isa AbstractField, args)
    rest = filter(x -> !(x isa AbstractField), args)
    raw_fields = ntuple(
        i -> fields[i] isa SpinorfieldEO ? fields[i].parent.U : fields[i].U, length(fields)
    )
    return (raw_fields..., rest...)
end

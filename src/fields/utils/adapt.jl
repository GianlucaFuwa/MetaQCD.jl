# INFO: This adapts an AbstractField such that it can be used within GPU
# kernels
function adapt_structure(to, u::AbstractField{B,T,M}) where {B,T,M}
    U = adapt_structure(to, u.U)
    # U = if u isa Spinorfield || u isa Colorfield || u isa Expfield
    #     adapt_structure(to, u.U.parent)
    # else
    #     ntuple(i -> adapt_structure(to, u.U[i].parent), Val(4))
    # end
    halos = if isnothing(u.halos)
        nothing
    else
        ntuple(i -> adapt_structure(to, u.halos[i]), Val(8))
    end
    sendbuf = if isnothing(u.sendbuf)
        nothing
    else
        ntuple(i -> adapt_structure(to, u.sendbuf[i]), Val(8))
    end
    topology = (
        halo_width = u.topology.halo_width,
        bulk_sites = u.topology.bulk_sites,
        bulk_sites_padded = u.topology.bulk_sites_padded,
        global_dims = u.topology.global_dims,
    )

    if u isa Gaugefield
        GA = gauge_action(u)
        N = nfloat(u)
        return Gaugefield{B,T,M,GA,N}(U, halos, sendbuf, topology, u.β, nothing)
    elseif u isa Spinorfield
        ND = num_dirac(u)
        return Spinorfield{B,T,M,ND}(U, halos, sendbuf, topology, nothing)
    elseif u isa MultiSpinorfield
        ND = num_dirac(u)
        return MultiSpinorfield{B,T,M,ND}(U, halos, sendbuf, topology, u.numspinors, nothing)
    elseif u isa Paulifield
        C = has_clover_term(u)
        return Paulifield{B,T,M,C}(U, halos, sendbuf, topology, u.csw, u.inverse, nothing)
    elseif u isa Colorfield
        return Colorfield{B,T,M}(U, halos, sendbuf, topology, nothing)
    elseif u isa Expfield
        return Expfield{B,T,M}(U, halos, sendbuf, topology, nothing)
    elseif u isa Tensorfield
        return Tensorfield{B,T,M}(U, halos, sendbuf, topology, nothing)
    end
end

# INFO: This converts u to a PtrArray pointing to the entries of u.U, meaning that we cant
# access any of the fields of u within the @batch loop
# @inline object_and_preserve(u::AbstractField) = object_and_preserve(u.U)
@generated function object_and_preserve(u::TF) where {T,M,TF<:AbstractField{CPU,T,M}}
    q = quote
        $(Expr(:meta, :inline))
    end

    Fieldtype = nameof(TF)
    fnames = fieldnames(TF)
    for name in fnames
        if name == :topology
            hw_expr = :(halo_width = u.topology.halo_width)
            bulk_expr = :(bulk_sites = u.topology.bulk_sites)
            bulk_pad_expr = :(bulk_sites_padded = u.topology.bulk_sites_padded)
            glob_expr = :(global_dims = u.topology.global_dims)
            push!(q.args, quote
                $(Symbol(:o_and_p_, name)) =
                    (($hw_expr, $bulk_expr, $bulk_pad_expr, $glob_expr), nothing)
            end)
        elseif name == :halo_valid
            push!(q.args, quote
                $(Symbol(:o_and_p_, name)) = (nothing, nothing)
            end)
        else
            push!(
                q.args,
                :($(Symbol(:o_and_p_, name)) = object_and_preserve(getfield(u, $(quot(name)))))
            )
        end
    end

    objects = Expr(:tuple)
    for name in fnames
        push!(objects.args, :($name = $(Symbol(:o_and_p_, name))[1]))
    end

    preserves = Expr(:tuple)
    for name in fnames
        push!(preserves.args, :($(Symbol(:o_and_p_, name))[2]))
    end

    push!(q.args, Expr(:(=), :objects, objects))
    push!(q.args, Expr(:(=), :preserves, preserves))

    q_field = Expr(:(=), :u_ptr)
    qu = if TF <: Gaugefield
        :(Gaugefield{CPU,T,M,gauge_action(u),nfloat(u)}($objects...))
    elseif TF <: Spinorfield || TF <: MultiSpinorfield
        :(Spinorfield{CPU,T,M,num_dirac(u)}($objects...))
    elseif TF <: Paulifield
        :(Paulifield{CPU,T,M,has_clover_term(u)}($objects...))
    else
        :($(Fieldtype){CPU,T,M}($objects...))
    end

    push!(q_field.args, qu)
    push!(q.args, q_field)
    push!(q.args, :(return u_ptr, preserves))
    return q
end


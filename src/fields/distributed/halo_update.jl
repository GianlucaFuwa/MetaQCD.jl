"""
    start_halo_update!(fields...; do_edges)

Start the update of halos or buffers of MPI-parallelized fields and return the Requests.
If `do_edges = Val(true)` edges and corners are also transferred via an extended face
propagation scheme. This means that after every dimension the requests have to be
completed and communication cannot be hidden behind computation.
"""
start_halo_update!(args...) = nothing

function start_halo_update!(
    fields::Vararg{AbstractMPIField,N}; do_edges::Val{DO_EDGES}=Val(true)
) where {N,DO_EDGES}
    requests = ntuple(Val(N)) do i
        start_halo_update!(fields[i], do_edges)
    end

    return requests
end

function start_halo_update!(u::AbstractMPIField, ::Val{do_edges}) where {do_edges}
    topology = u.topology
    comm_cart = topology.comm_cart
    comm_instance = mpi_comm_instance()
    halo_sites = topology.halo_sites
    border_sites = topology.border_sites

    requests = Utils.MPI.Request[]

    for dim in 1:4
        dim_requests = Utils.MPI.Request[]
        prev_neighbor, next_neighbor = mpi_cart_shift(comm_cart, dim-1, 1)
        prev_sites_from, next_sites_from = border_sites[dim]
        prev_sites_to, next_sites_to = halo_sites[dim]

        if prev_neighbor == next_neighbor == mpi_myrank(comm_instance)
            copyto!(u, u, next_sites_to, prev_sites_from)
            copyto!(u, u, prev_sites_to, next_sites_from)
        else
            # INFO: Here, `view` is defined such that it automatically references all four
            # directions `μ`, and we don't have to include it as an argument
            send_buf_prev = create_sendbuf!(u, prev_sites_from, dim, 1)
            send_buf_next = create_sendbuf!(u, next_sites_from, dim, 2)
            recv_buf_prev = u.halos[2(dim-1) + 1].parent # INFO: MPI.Buffer not overloaded for OffsetArrays
            recv_buf_next = u.halos[2(dim-1) + 2].parent

            push!(
                dim_requests,
                mpi_irecv!(recv_buf_prev, comm_cart; source=prev_neighbor, tag=1),
                mpi_isend(send_buf_next, comm_cart; dest=next_neighbor, tag=1),
                mpi_irecv!(recv_buf_next, comm_cart; source=next_neighbor, tag=2),
                mpi_isend(send_buf_prev, comm_cart; dest=prev_neighbor, tag=2)
            )
        end

        # If we care about edges and corners we have to wait here, because then the
        # elements exchanged in the next dimension depend on the previous ones
        # otherwise we dont wait, because the faces are always correctly exchanged
        if do_edges
            finalize_halo_update!(dim_requests)
            mpi_barrier()
        else
            push!(requests, dim_requests...)
        end
    end

    return requests
end

"""
    finalize_halo_update!(reqs...)

Wait on all started halo updates in `reqs` to finish.
"""
finalize_halo_update!(args...) = nothing

function finalize_halo_update!(reqs::Vararg{Vector{Utils.MPI.Request},N}) where N
    for i in 1:N
        finalize_halo_update!(reqs[i])
    end
    return nothing
end

function finalize_halo_update!(reqs::Vector{Utils.MPI.Request})
    mpi_waitall(reqs)
    return nothing
end

"""
    update_halo!(fields...)

Perform a complete halo exchange. Use this function when communication cannot be hidden.
(E.g., when edges and corners need to be exchanges)
"""
update_halo!(args...) = nothing

function update_halo!(fields::Vararg{AbstractMPIField,N}) where N
    start_halo_update!(fields...; do_edges=Val(true))
    return nothing
end

"""
    @hide_communication(fields..., stencil_size, ex)

Hide communication behind computation by splitting all loops over the bulk into a loop
over the inner bulk, i.e., those sites that dont depend on halos and the outer bulk,
i.e., those that do.
"""
macro hide_communication(args...) # fields_and_stencil, block
    quote
        $(esc(_hide_communication(args...)))
    end
end

function _hide_communication(args...)
    # Extract fields and stencil size
    fields = args[1:end-2]
    __stencil_size__ = args[end-1]
    @assert __stencil_size__ isa Int64
    block = args[end]
    @assert block isa Expr

    # Loop to find the @batch for loop inside the block
    idcs = findall(s->contains(string(s), string("@batch")), block.args)
    @assert length(idcs) == 1 "No @batch loop found, couldn't hide communication"
    idx = idcs[1]
    loop = block.args[idx]
    red_idx = findfirst(s->contains(string(s), string("reduction")), loop.args)
    has_reduction = !isnothing(red_idx)
    red_sym = has_reduction ? loop.args[red_idx] : :()
    raw_loop = loop.args[end]

    # Extract loop inner and indexed field
    @capture(raw_loop, (for __site_ = eachindex(ξ_) inner_ end))
    @assert ξ in fields "Field from eachindex(...) has to be @hide_communications argument"

    inner_loop = Expr(:for, :(site = __inner_bulk__), inner)
    inner_call = Expr(:macrocall, Symbol("@batch"), red_sym, inner_loop)

    outer_loop = Expr(:for, :(site = __outer_bulk__), inner)
    outer_call = Expr(:macrocall, Symbol("@batch"), red_sym, outer_loop)

    block.args[idx] = quote
        if !is_distributed($ξ)
            $loop
        else
            local __exchange_requests__ = start_halo_update!($(fields)...)
            local __inner_bulk__ = shrink_bulk(eachindex($ξ), $__stencil_size__)
            local __outer_bulk__ = $ξ.topology.border_sites[$(__stencil_size__ + 1)]

            try
                $inner_call
            finally
                finalize_halo_update!(__exchange_requests__)
            end

            $outer_call
        end
    end

    return block
end

@inline function shrink_bulk(bulk, stencil_radius)
    new_ranges = ntuple(Val(4)) do i
        irange = bulk.indices[i]
        new_first = first(irange) + stencil_radius
        new_last = last(irange) - stencil_radius
        @assert new_first <= new_last
        range(new_first, new_last)
    end

    return CartesianIndices(new_ranges)
end

@inline distributed_reduce(var, ::Any, ::AbstractField) = var

@inline function distributed_reduce(var, op, u::AbstractMPIField)
    return mpi_allreduce(var, op, u.topology.comm_cart)
end

proc_offset(u::AbstractField) = 0

@inline function proc_offset(u::AbstractMPIField)
    offset = map(r -> first(r) - 1, u.topology.bulk_sites.indices)
    return offset
end

function find_batch_loop(ex)
    idcs = findall(s->contains(string(s), string("@batch")), ex.args[2].args)
    @assert length(idcs) == 1 "No @batch loop found, couldn't hide communication"
    idx = idcs[1]

    loop = ex.args[2].args[idx]
    rawloop = loop.args[end]
    return loop, rawloop, idx
end

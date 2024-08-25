distributed_reduce(var, ::Any, ::AbstractField{B,T,false}) where {B,T} = var

@inline function distributed_reduce(var, op, u::AbstractField{B,T,true}) where {B,T}
    return mpi_allreduce(var, op, u.comm_cart)
end

proc_offset(u::AbstractField{B,T,false}) where {B,T} = 0

@inline function proc_offset(u::AbstractField{B,T,true}) where {B,T}
    NX, NY, NZ, _ = global_dims(u)
    my_NX, my_NY, my_NZ, my_NT = local_dims(u)
    myrank_x, myrank_y, myrank_z, myrank_t = u.myrank_cart
    
    offset_x = myrank_x * my_NX
    offset_y = myrank_y * NX * my_NY
    offset_z = myrank_z * NX * NY * my_NZ
    offset_t = myrank_t * NX * NY * NZ * my_NT

    offset = offset_x + offset_y + offset_z + offset_t 
    return offset
end

@inline halo_dims(u::AbstractField, dir) = halo_dims(local_dims(u), u.pad, dir)

@inline function halo_dims(loc_dims, pad, dir)
    my_NX, my_NY, my_NZ, my_NT = loc_dims

    if dir == 1
        return (pad, my_NY, my_NZ, my_NT)
    elseif dir == 2
        return (my_NX, pad, my_NZ, my_NT)
    elseif dir == 3
        return (my_NX, my_NY, pad, my_NT)
    elseif dir == 4
        return (my_NX, my_NY, my_NZ, pad)
    else
        throw(AssertionError("dir has to be between 1 and 4"))
    end
end

@inline function halo_sites(u::AbstractField, dim)
    my_NX, my_NY, my_NZ, my_NT = local_dims(u)
    pad = u.pad

    if dim == 1
        prev = CartesianIndices((1:pad, my_NY, my_NZ, my_NT))
        next = CartesianIndices((my_NX+pad:my_NX+2pad, my_NY, my_NZ, my_NT))
        return prev, next
    elseif dim == 2
        prev = CartesianIndices((my_NX, 1:pad, my_NZ, my_NT))
        next = CartesianIndices((my_NX, my_NY+pad:my_NY+2pad, my_NZ, my_NT))
        return prev, next
    elseif dim == 3
        prev = CartesianIndices((my_NX, my_NY, 1:pad, my_NT))
        next = CartesianIndices((my_NX, my_NY, my_NZ+pad:my_NZ+2pad, my_NT))
        return prev, next
    elseif dim == 4
        prev = CartesianIndices((my_NX, my_NY, my_NZ, 1:pad))
        next = CartesianIndices((my_NX, my_NY, my_NZ, my_NT+pad:my_NT+2pad))
        return prev, next
    else
        throw(AssertionError("dim has to be between 1 and 4"))
    end
end

@inline function bulk_boundary_sites(u::AbstractField, dim)
    my_NX, my_NY, my_NZ, my_NT = local_dims(u)
    pad = u.pad

    if dim == 1
        prev = CartesianIndices((1+pad:2pad, my_NY, my_NZ, my_NT))
        next = CartesianIndices((my_NX-pad:my_NX, my_NY, my_NZ, my_NT))
        return prev, next
    elseif dim == 2
        prev = CartesianIndices((my_NX, 1+pad:2pad, my_NZ, my_NT))
        next = CartesianIndices((my_NX, my_NY-pad:my_NY, my_NZ, my_NT))
        return prev, next
    elseif dim == 3
        prev = CartesianIndices((my_NX, my_NY, 1+pad:2pad, my_NT))
        next = CartesianIndices((my_NX, my_NY, my_NZ-pad:my_NZ, my_NT))
        return prev, next
    elseif dim == 4
        prev = CartesianIndices((my_NX, my_NY, my_NZ, 1+pad:2pad))
        next = CartesianIndices((my_NX, my_NY, my_NZ, my_NT-pad:my_NT))
        return prev, next
    else
        throw(AssertionError("dim has to be between 1 and 4"))
    end
end

update_halo!(::AbstractField{B,T,false}) where {B,T} = nothing

function update_halo!(u::AbstractField{B,T,true}) where {B,T}
    comm_cart = u.comm_cart
    send_buf = u.halo_sendbuf
    recv_buf = u.halo_recvbuf

    for dir in 1:4
        requests = mpi_multirequest(4)
        prev_neighbor, next_neighbor = mpi_cart_shift(comm_cart, dir-1, 1)
        # Source sites (bulk boundary):
        prev_sites_from, next_sites_from = bulk_boundary_sites(u, dir)
        # Sink sites (halo):
        prev_sites_to, next_sites_to = halo_sites(u, dir)

        halo_NX, halo_NY, halo_NZ, _ = halo_dims(u, dir)

        # send - negative direction
        start1 = prev_sites_from[1]

        for site in prev_sites_from
            for μ in 1:4
                i = cartesian_to_linear(site, halo_NX, halo_NY, halo_NZ, start1)
                send_buf[μ, i] = u[μ, site]
            end
        end

        mpi_isend(send_buf, comm_cart, requests[1]; dest=prev_neighbor)

        # send - positive direction
        start2 = next_sites_from[1]

        for site in next_sites_from
            for μ in 1:4
                i = cartesian_to_linear(site, halo_NX, halo_NY, halo_NZ, start2)
                send_buf[μ, i] = u[μ, site]
            end
        end

        mpi_isend(send_buf, comm_cart, requests[2]; dest=next_neighbor)

        # recv - negative direction
        mpi_irecv!(recv_buf, comm_cart, requests[3]; source=next_neighbor)
        start3 = prev_sites_to[1]

        for site in prev_sites_to
            for μ in 1:4
                i = cartesian_to_linear(site, halo_NX, halo_NY, halo_NZ, start3)
                u[μ, site] = recv_buf[μ, i]
            end
        end

        # recv - positive direction
        mpi_irecv!(recv_buf, comm_cart, requests[4]; source=prev_neighbor)
        start4 = next_sites_to[1]

        for site in next_sites_to
            for μ in 1:4
                i = cartesian_to_linear(site, halo_NX, halo_NY, halo_NZ, start4)
                u[μ, site] = recv_buf[μ, i]
            end
        end

        mpi_waitall(requests)
    end

    return nothing
end

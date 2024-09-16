"""
    FieldTopology(numprocs_cart, halo_width, global_dims)

Given the number of processes in each dimension as a tuple `numprocs_cart`, the `halo_width`
and he global dimensions of the field `global_dims`, create a container for all information
related to the (MPI-)Topology of an `AbstractField`, such as its global and local
dimensions, `CartesianIndices` for iterating over halo or bulk sites and the site rannge of
the global field that the local partition contains.
"""
struct FieldTopology
    comm_cart::Utils.MPI.Comm

    numprocs::Int64 # Number of processes in comm
    numprocs_cart::NTuple{4,Int64} # Number of processes in comm per dimension
    myrank_cart::NTuple{4,Int64} # Rank of current process in cartesian coords

    halo_width::Int64
    global_dims::NTuple{4,Int64} # Dimensions of global field
    local_dims::NTuple{4,Int64} # Dimensions of local partition
    halo_dims::Vector{NTuple{4,Int64}} # Dimensions of halo regions

    bulk_sites::CartesianIndices{4} # Sites in partition that belong to the bulk
    halo_sites::Vector{NTuple{2,CartesianIndices{4}}} # Sites in partition that belong to halo regions (2 per dim)
    border_sites::Vector{NTuple{2,CartesianIndices{4}}} # Sites in bulk that belong to border regions (2 per dim)

    local_ranges::NTuple{4,UnitRange{Int64}} # Site-range of global field that the local partition covers

    global_volume::Int64 # Number of sites in global field
    local_volume::Int64 # Number of sites in local partition
    function FieldTopology(numprocs_cart, halo_width, global_dims)
        @assert global_dims .% numprocs_cart == (0, 0, 0, 0) "Lattice size must be divisible by number of processes per dimension"
        @assert minimum(global_dims ./ numprocs_cart) >= halo_width "Halo must not be wider than the bulk"
        comm_cart = mpi_cart_create(numprocs_cart; periodic=map(_->true, numprocs_cart))

        numprocs = prod(numprocs_cart)
        myrank_cart = (mpi_cart_coords(comm_cart, mpi_myrank())...,)

        local_dims = global_dims .÷ numprocs_cart
        halo_dims = calc_halo_dims(local_dims, halo_width)
        
        bulk_sites = calc_bulk_sites(local_dims, halo_width)
        halo_sites = calc_halo_sites(bulk_sites, local_dims, halo_width)
        border_sites = calc_border_sites(bulk_sites, local_dims, halo_width)

        local_ranges = ntuple(Val(4)) do i
            (myrank_cart[i] * local_dims[i] + 1):((myrank_cart[i]+1) * local_dims[i])
        end

        global_volume = prod(global_dims)
        local_volume = prod(local_dims)
        return new(
            comm_cart, numprocs, numprocs_cart, myrank_cart,
            halo_width, global_dims, local_dims, halo_dims,
            bulk_sites, halo_sites, border_sites,
            local_ranges, global_volume, local_volume,
        )
    end
end

@inline function calc_halo_dims(local_dims, halo_width)
    nx, ny, nz, nt = local_dims
    halo_dims = Vector{NTuple{4,Int64}}(undef, 4)

    halo_dims[1] = (halo_width, ny, nz, nt)
    halo_dims[2] = (nx, halo_width, nz, nt)
    halo_dims[3] = (nx, ny, halo_width, nt)
    halo_dims[4] = (nx, ny, nz, halo_width)
    return halo_dims
end

@inline function calc_bulk_sites(local_dims, halo_width)
    iend = local_dims .+ halo_width

    ranges = ntuple(Val(4)) do i
        range(1 + halo_width, iend[i])
    end

    return CartesianIndices(ranges)
end

function calc_halo_sites(bulk_sites, local_dims, halo_width)
    halo_sites = Vector{NTuple{2,CartesianIndices{4}}}(undef, 4)

    for dim in 1:4
        prev_tup = ntuple(Val(4)) do i 
            if i == dim
                range(1, halo_width)
            elseif i < dim
                # INFO: This is for correct exchange of corners:
                range(1, local_dims[i] + 2halo_width)
            else
                bulk_sites.indices[i]
            end
        end

        next_tup = ntuple(Val(4)) do i
            ni = local_dims[i]
            hw = halo_width
            if i == dim
                range(1+ni+hw, ni+2hw)
            elseif i < dim
                # INFO: This is for correct exchange of corners:
                range(1, local_dims[i] + 2hw)
            else
                bulk_sites.indices[i]
            end
        end

        prev = CartesianIndices(prev_tup)
        next = CartesianIndices(next_tup)
        halo_sites[dim] = (prev, next)
    end

    return halo_sites
end

function calc_border_sites(bulk_sites, local_dims, halo_width)
    border_sites = Vector{NTuple{2,CartesianIndices{4}}}(undef, 4)

    for dim in 1:4
        prev_tup = ntuple(Val(4)) do i 
            hw = halo_width
            if i == dim
                range(1+hw, 2hw)
            elseif i < dim
                # INFO: This is for correct exchange of corners:
                range(1, local_dims[i] + 2hw)
            else
                bulk_sites.indices[i]
            end
        end

        next_tup = ntuple(Val(4)) do i
            ni = local_dims[i]
            if i == dim
                range(1+ni, ni+halo_width)
            elseif i < dim
                # INFO: This is for correct exchange of corners:
                range(1, local_dims[i] + 2halo_width)
            else
                bulk_sites.indices[i]
            end
        end

        prev = CartesianIndices(prev_tup)
        next = CartesianIndices(next_tup)
        border_sites[dim] = (prev, next)
    end

    return border_sites
end

@inline distributed_reduce(var, ::Any, ::AbstractField) = var

@inline function distributed_reduce(var, op, u::AbstractMPIField)
    return mpi_allreduce(var, op, u.topology.comm_cart)
end

proc_offset(u::AbstractField) = 0

@inline function proc_offset(u::AbstractMPIField)
    offset = map(r -> first(r) - 1, local_ranges(u))
    return offset
end

Utils.update_halo!(::AbstractField) = nothing

function Utils.update_halo!(u::AbstractMPIField)
    topology = u.topology
    comm_cart = topology.comm_cart
    border_sites = topology.border_sites
    halo_sites = topology.halo_sites

    for dim in 1:4
        prev_neighbor, next_neighbor = mpi_cart_shift(comm_cart, dim-1, 1)
        prev_sites_from, next_sites_from = border_sites[dim]
        prev_sites_to, next_sites_to = halo_sites[dim]

        if prev_neighbor == next_neighbor == mpi_myrank()
            view(u, next_sites_to) .= view(u, prev_sites_from)
            view(u, prev_sites_to) .= view(u, next_sites_from)
        else
            requests = mpi_multirequest(4)

            # Use references of the links themselves as buffers
            # INFO: Here, `view` is defined such that it automatically references all four
            # directions `μ`, and we don't have to include it as an argument
            send_buf_prev = view(u, prev_sites_from)
            send_buf_next = view(u, next_sites_from)
            recv_buf_prev = view(u, prev_sites_to)
            recv_buf_next = view(u, next_sites_to)

            mpi_irecv!(recv_buf_prev, comm_cart, requests[1]; source=prev_neighbor, tag=1)
            mpi_irecv!(recv_buf_next, comm_cart, requests[2]; source=next_neighbor, tag=2)
            mpi_isend(send_buf_prev, comm_cart, requests[3]; dest=prev_neighbor, tag=2)
            mpi_isend(send_buf_next, comm_cart, requests[4]; dest=next_neighbor, tag=1)

            mpi_waitall(requests)
        end
    end

    mpi_barrier(comm_cart)
    return nothing
end

update_halo_eo!(::AbstractField) = nothing

function update_halo_eo!(u::AbstractMPIField)
    topology = u.topology
    comm_cart = topology.comm_cart
    border_sites = topology.border_sites
    halo_sites = topology.halo_sites

    for dim in 1:4
        prev_neighbor, next_neighbor = mpi_cart_shift(comm_cart, dim-1, 1)
        prev_sites_from, next_sites_from = border_sites[dim]
        prev_sites_to, next_sites_to = halo_sites[dim]

        if prev_neighbor == next_neighbor == mpi_myrank()
            view(u, next_sites_to) .= view(u, prev_sites_from)
            view(u, prev_sites_to) .= view(u, next_sites_from)
        else
            requests = mpi_multirequest(4)

            # Use references of the links themselves as buffers
            # INFO: Here, `view` is defined such that it automatically references all four
            # directions `μ`, and we don't have to include it as an argument
            send_buf_prev = view(u, prev_sites_from)
            send_buf_next = view(u, next_sites_from)
            recv_buf_prev = view(u, prev_sites_to)
            recv_buf_next = view(u, next_sites_to)

            mpi_irecv!(recv_buf_prev, comm_cart, requests[1]; source=prev_neighbor, tag=1)
            mpi_irecv!(recv_buf_next, comm_cart, requests[2]; source=next_neighbor, tag=2)
            mpi_isend(send_buf_prev, comm_cart, requests[3]; dest=prev_neighbor, tag=2)
            mpi_isend(send_buf_next, comm_cart, requests[4]; dest=next_neighbor, tag=1)

            mpi_waitall(requests)
        end
    end

    mpi_barrier(comm_cart)
    return nothing
end

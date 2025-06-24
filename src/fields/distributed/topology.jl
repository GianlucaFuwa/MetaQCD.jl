# INFO: Convenience structs for halo and border sites
const ContiguousExchangeSites = NTuple{
    4,NTuple{2,CartesianIndices{4,NTuple{4,UnitRange{Int64}}}}
} # [dim][side][sites]

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
    local_dims::NTuple{4,Int64} # Dimensions of local bulk

    origin::CartesianIndex{4}
    # Sites in partition that belong to the bulk
    bulk_sites::CartesianIndices{4,NTuple{4,UnitRange{Int64}}}
    bulk_sites_padded::CartesianIndices{4,NTuple{4,UnitRange{Int64}}}
    # Sites in partition that belong to halo regions (forward and backward per dim)
    halo_sites::ContiguousExchangeSites
    # Sites in bulk that belong to border regions (forward and backward per dim)
    border_sites::ContiguousExchangeSites # one for each stencil size up to halo_width

    global_volume::Int64 # Number of sites in global field
    local_volume::Int64 # Number of sites in local partition
    function FieldTopology(numprocs_cart, halo_width, global_dims)
        @assert global_dims .% numprocs_cart == (0, 0, 0, 0) """
        Lattice size must be divisible by number of processes per dimension
        """
        @assert minimum(global_dims ./ numprocs_cart) >= halo_width """
        Halo must not be wider than the bulk
        """
        comm_cart = mpi_cart_create(
            mpi_comm_instance(), numprocs_cart; periodic=map(_->true, numprocs_cart)
        )

        numprocs = prod(numprocs_cart)
        myrank_cart = numprocs == 1 ? (0, 0, 0, 0) : (mpi_cart_coords(comm_cart)...,)

        local_dims = global_dims .÷ numprocs_cart

        global_sites = CartesianIndices(ntuple(Val(4)) do i
            (myrank_cart[i] * local_dims[i] + 1):((myrank_cart[i]+1) * local_dims[i])
        end)

        origin = global_sites[1]
        bulk_sites = global_sites
        bulk_sites_padded = CartesianIndices(ntuple(Val(4)) do i
            bulk_i = bulk_sites.indices[i]
            hw = numprocs_cart[i] == 1 ? 0 : halo_width
            range(first(bulk_i) - hw, last(bulk_i) + hw)
        end)
        halo_sites = calc_halo_sites(bulk_sites, local_dims, halo_width)
        border_sites = calc_border_sites(bulk_sites, local_dims, halo_width)

        global_volume = prod(global_dims)
        local_volume = prod(local_dims)
        return new(
            comm_cart, numprocs, numprocs_cart, myrank_cart,
            halo_width, global_dims, local_dims,
            origin, bulk_sites, bulk_sites_padded, halo_sites, border_sites,
            global_volume, local_volume,
        )
    end
end

function calc_halo_sites(bulk_sites, local_dims, halo_width)
    origin = bulk_sites[1]

    halo_sites = ntuple(Val(4)) do dim
        prev_tup = ntuple(Val(4)) do i 
            if i == dim
                range(origin[i]-halo_width, origin[i]-1)
            elseif i < dim
                # INFO: This is for correct exchange of corners:
                range(
                    origin[i]-halo_width,
                    origin[i]+local_dims[i]+halo_width-1,
                )
            else
                bulk_sites.indices[i]
            end
        end

        next_tup = ntuple(Val(4)) do i
            if i == dim
                range(
                    origin[i]+local_dims[i],
                    origin[i]+local_dims[i]+halo_width-1,
                )
            elseif i < dim
                # INFO: This is for correct exchange of corners:
                range(
                    origin[i]-halo_width,
                    origin[i]+local_dims[i]+halo_width-1,
                )
            else
                bulk_sites.indices[i]
            end
        end

        prev = CartesianIndices(prev_tup)
        next = CartesianIndices(next_tup)
        (prev, next)
    end

    return halo_sites
end

function calc_border_sites(bulk_sites, local_dims, halo_width)
    origin = bulk_sites[1]

    border_sites = ntuple(Val(4)) do dim
        prev_tup = ntuple(Val(4)) do i 
            if i == dim
                range(origin[i], origin[i]+halo_width-1)
            elseif i < dim
                # INFO: This is for correct exchange of corners:
                range(
                    origin[i]-halo_width,
                    origin[i]+local_dims[i]+halo_width-1,
                )
            else
                bulk_sites.indices[i]
            end
        end

        next_tup = ntuple(Val(4)) do i
            if i == dim
                range(
                    origin[i]+local_dims[i]-halo_width,
                    origin[i]+local_dims[i]-1,
                )
            elseif i < dim
                # INFO: This is for correct exchange of corners:
                range(
                    origin[i]-halo_width,
                    origin[i]+local_dims[i]+halo_width-1,
                )
            else
                bulk_sites.indices[i]
            end
        end

        prev = CartesianIndices(prev_tup)
        next = CartesianIndices(next_tup)
        (prev, next)
    end

    return border_sites
end


function save_field(::BridgeFormat, u::AbstractField{B,T,true}, filename) where {B,T}
    return save_field_mpi(u, filename)
end

function save_field_mpi(u::AbstractField{B,T}, filename, args...) where {B,T}
    fp = Utils.MPI.File.open(u.topology.comm_cart, filename; write=true)
    set_view!(fp, u, eltype(u))
    Utils.MPI.File.write_all(fp, device_to_host(u.U.parent, B))
    Utils.MPI.File.close(fp)
    mpi_barrier(u.topology.comm_cart)
    return nothing
end

function load_field!(::BridgeFormat, u::AbstractField{B,T,true}, filename) where {B,T}
    return load_field_mpi!(u, filename)
end

function load_field_mpi!(u::AbstractField{B,T,M}, filename) where {B,T,M}
    fp = Utils.MPI.File.open(u.topology.comm_cart, filename; read=true)
    etype = eltype(u)
    set_view!(fp, u, etype)

    inner_len = if u isa Spinorfield || u isa SpinorfieldEO || u isa Paulifield
        1
    elseif u isa Tensorfield
        16
    elseif u isa MultiSpinorfield
        u.numspinors
    else
        4
    end

    tmp = zeros(etype, inner_len * u.topology.local_volume)
    Utils.MPI.File.read_all!(fp, tmp)
    tmp = device_to_host(tmp, B)

    ind = allindices(u)
    itr = eachindex(IndexLinear(), ind)

    parallelfor(itr, B, Val(M), Val(false), (), (u,), (u,)) do i, u
        μsite = ind[i]
        u[μsite] = tmp[i]
    end

    Utils.MPI.File.close(fp)
    mpi_barrier(u.topology.comm_cart)
    return nothing
end

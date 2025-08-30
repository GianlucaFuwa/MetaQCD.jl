function save_field(::BridgeFormat, u::AbstractField{B,T,true}, filename) where {B,T}
    return save_field_mpi(u, filename)
end

function save_field_mpi(u::Gaugefield{B,T}, filename, args...) where {B,T}
    filename != "" || return nothing
    fp = Utils.MPI.File.open(u.topology.comm_cart, filename; write=true)
    etype = SMatrix{3,3,ComplexF64,9}
    U = if B == CPU
        u.U.parent
    else
        tmp = bzeros(B(), etype, 4, u.topology.local_volume)
        sites = eachindex(u)
        parallelfor(1:length(sites), B, Val(false), (), (), (u,)) do i, (u,)
            site = sites[i]
            tmp[1, i] = u[1, site]
            tmp[2, i] = u[2, site]
            tmp[3, i] = u[3, site]
            tmp[4, i] = u[4, site]
        end
        Array(tmp)
    end
    set_view!(fp, u, etype)
    Utils.MPI.File.write_all(fp, U)
    Utils.MPI.File.close(fp)
    mpi_barrier(u.topology.comm_cart)
    return nothing
end

function load_field!(::BridgeFormat, u::AbstractField{B,T,true}, filename) where {B,T}
    return load_field_mpi!(u, filename)
end

function load_field_mpi!(u::Gaugefield{B,T,M}, filename) where {B,T,M}
    convert_fun = B == CPU ? identity : array_type(B)
    fp = Utils.MPI.File.open(u.topology.comm_cart, filename; read=true)
    # TODO: etype should be smatrix/svector also for B<:GPU
    # need to convert in parallelfor loop if T!=Float64
    etype = SMatrix{3,3,ComplexF64,9}
    set_view!(fp, u, etype)

    _tmp = zeros(etype, 4, u.topology.local_volume)
    Utils.MPI.File.read_all!(fp, _tmp)
    tmp = convert_fun(_tmp)
    sites = eachindex(u)

    parallelfor(1:length(sites), B, Val(M), Val(false), (), (u,), (u,)) do i, (u,)
        site = sites[i]
        u[1, site] = tmp[1, i]
        u[2, site] = tmp[2, i]
        u[3, site] = tmp[3, i]
        u[4, site] = tmp[4, i]
    end

    Utils.MPI.File.close(fp)
    mpi_barrier(u.topology.comm_cart)
    return nothing
end

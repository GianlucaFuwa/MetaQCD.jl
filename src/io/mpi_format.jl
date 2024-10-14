function save_config(::BridgeFormat, U::Gaugefield{B,T,true}, filename, args...) where {B,T}
    @assert U.U isa Array
    fp = Utils.MPI.File.open(U.topology.comm_cart, filename; write=true)
    set_view!(fp, U, SMatrix{3,3,Complex{T},9})
    Utils.MPI.File.write_all(fp, view(U.U, :, U.topology.local_ranges...))
    Utils.MPI.File.close(fp)
    mpi_barrier(U.topology.comm_cart)
    return nothing
end

function load_config!(::BridgeFormat, U::Gaugefield{B,T,true}, filename) where {B,T}
    @assert U.U isa Array
    fp = Utils.MPI.File.open(U.topology.comm_cart, filename; read=true)

    set_view!(fp, U, SMatrix{3,3,ComplexF64,9})

    tmp = Vector{SMatrix{3,3,ComplexF64,9}}(undef, 4U.topology.local_volume)
    Utils.MPI.File.read_all!(fp, tmp)
    i = 1

    for site in eachindex(U)
        for μ in 1:4
            U[μ, site] = tmp[i]
            i += 1
        end
    end

    Utils.MPI.File.close(fp)
    mpi_barrier(U.topology.comm_cart)
    update_halo!(U)
    return nothing
end

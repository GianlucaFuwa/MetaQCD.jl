# TODO: Use new Topology with OffsetArrays
function save_config(::BridgeFormat, U::Gaugefield{CPU,T,true}, filename, args...) where {T}
    fp = Utils.MPI.File.open(U.topology.comm_cart, filename; write=true)
    set_view!(fp, U, SMatrix{3,3,Complex{T},9})
    Utils.MPI.File.write_all(fp, U.U)
    Utils.MPI.File.close(fp)
    mpi_barrier(U.topology.comm_cart)
    return nothing
end

function save_config(::Bool, U::Gaugefield{CPU,T}, filename, args...) where {T}
    fp = Utils.MPI.File.open(U.topology.comm_cart, filename; write=true)
    set_view!(fp, U, SMatrix{3,3,Complex{T},9})
    Utils.MPI.File.write_all(fp, U.U)
    Utils.MPI.File.close(fp)
    mpi_barrier(U.topology.comm_cart)
    return nothing
end

function load_config!(::BridgeFormat, U::Gaugefield{CPU,T,true}, filename) where {T}
    fp = Utils.MPI.File.open(U.topology.comm_cart, filename; read=true)
    set_view!(fp, U, SMatrix{3,3,ComplexF64,9})

    tmp = zeros(SMatrix{3,3,ComplexF64,9}, 4U.topology.local_volume)
    Utils.MPI.File.read_all!(fp, tmp)
    i = 1

    # TODO: for GPU
    for site in eachindex(U)
        for μ in 1:4
            @assert tmp[i] != zero(SMatrix{3,3,ComplexF64,9})
            U[μ, site] = tmp[i]
            i += 1
        end
    end

    Utils.MPI.File.close(fp)
    mpi_barrier(U.topology.comm_cart)
    return nothing
end

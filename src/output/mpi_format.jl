function save_config(::BridgeFormat, U, filename, ::Bool)
    @assert U.U isa Array
    NX, NY, NZ, NT = (U.NX, U.NY, U.NZ, U.NT)
    fp = Utils.MPI.File.open(mpi_comm(), filename; write=true)

    offset = 0
    for it in 1:NT
        for iz in 1:NZ
            for iy in 1:NY
                for ix in 1:NX
                    for μ in 1:4
                        for a in 1:3
                            for b in 1:3
                                rvalue = real(U[μ, ix, iy, iz, it][a, b])
                                Utils.MPI.File.write_at(fp, offset, rvalue)
                                offset += sizeof(rvalue)
                                ivalue = imag(U[μ, ix, iy, iz, it][a, b])
                                Utils.MPI.File.write_at(fp, offset, ivalue)
                                offset += sizeof(ivalue)
                            end
                        end
                    end
                end
            end
        end
    end

    Utils.MPI.File.close(fp)
    return nothing
end

function load_config!(::BridgeFormat, U, filename, ::Bool, T::Type{<:Real}=Float64)
    @assert U.U isa Array
    NX, NY, NZ, NT = (U.NX, U.NY, U.NZ, U.NT)
    my_NX, my_NY, my_NZ, my_NT = (U.my_NX, U.my_NY, U.my_NZ, U.my_NT)
    myrank_x, myrank_y, myrank_z, myrank_t = U.myrank_cart
    pad = U.pad

    num_el = 4 * 18 # 4 directions, 18 floats

    offset_x = myrank_x * my_NX * num_el
    offset_y = myrank_y * my_NY * num_el
    offset_z = myrank_z * my_NZ * num_el
    offset_t = myrank_t * my_NT * num_el

    offset = offset_x + offset_y + offset_z + offset_t 
    fp = Utils.MPI.File.open(U.comm_cart, filename; read=true)

    etype = Utils.MPI.Datatype(T)
    subarray = Utils.MPI.Types.create_subarray(
        (NX * num_el, NY * num_el, NZ * num_el, NT * num_el), 
        (my_NX * num_el, my_NY * num_el, my_NZ * num_el, my_NT * num_el),
        (offset_x, offset_y, offset_z, offset_t),
        etype,
    )
    Utils.MPI.Types.commit!(subarray)
    Utils.MPI.File.set_view!(fp, offset, etype, subarray, "native")

    tmp = Vector{Float64}(undef, U.my_NV * 18 * 4)
    Utils.MPI.File.read_all!(fp, tmp)

    i = 1
    for it in 1+pad:my_NT+pad
        for iz in 1+pad:my_NZ+pad
            for iy in 1+pad:my_NY+pad
                for ix in 1+pad:my_NX+pad
                    for μ in 1:4
                        link = @MMatrix zeros(Complex{T}, 3, 3)
                        for a in 1:3
                            for b in 1:3
                                rvalue = tmp[i]
                                i += 1
                                ivalue = tmp[i]
                                i += 1
                                link[a, b] = rvalue + im * ivalue
                            end
                        end
                        U[μ, ix, iy, iz, it] = SMatrix{3,3,ComplexF64,9}(link)
                    end
                end
            end
        end
    end

    Utils.MPI.File.close(fp)
    @show U.myrank_cart
    if U.myrank_cart == (0, 0, 0, 0)
        @show U[1, 1+pad, 1+pad, 1+pad, 1+pad]
    elseif U.myrank_cart == (0, 0, 1, 1)
        @show U[1, 4+pad, 4+pad, 2+pad, 2+pad]
    end
    mpi_barrier(U.comm_cart)
    return nothing
end

function set_view!(fp, U, offset, ::Type{T}; infokws...) where {T}
    etype = Utils.MPI.Datatype(T)
    filetype = create_filetype(U, T)
    datarep = "native"
    Utils.MPI.File.set_view!(fp, offset, etype, filetype, datarep; infokws...)
    return nothing
end

function create_filetype(U, ::Type{T}) where {T}
    global_dims = (4, U.NX, U.NY, U.NZ, U.NT)
    local_dims = (4, U.my_NX, U.my_NY, U.my_NZ, U.my_NT)
    local_starts = (1, U.myrank_cart...) .* local_dims
    local_range = (1:4, ) 
    offsets = map(r -> (first(r) - 1) * 18, )
    oldtype = Utils.MPI.Datatype(T)
    ftype = Utils.MPI.Types.create_subarray(global_dims, local_dims, offsets, oldtype)
    Utils.MPI.Types.commit!(ftype)
    return ftype
end

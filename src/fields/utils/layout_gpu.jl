function create_gpu_layout(struct_name)
    ldims, tuple_len = if struct_name == :Gaugefield
        :(N == 18 ? 9 : 3, topology.local_dims_padded..., 4), 4
    elseif struct_name == :Spinorfield
        :(ND == 1 ? 3 : 6, topology.local_dims_padded...), 0
    elseif struct_name == :MultiSpinorfield
        :(ND == 1 ? 3 : 6, topology.local_dims_padded..., numspinors), :numspinors
    elseif struct_name == :Tensorfield
        :(9, topology.local_dims_padded..., 6), 6
    elseif struct_name == :Colorfield
        :(9, topology.local_dims_padded..., 4), 0
    elseif struct_name == :Expfield
        :(topology.local_dims_padded..., 4), 0
    elseif struct_name == :Paulifield
        quote
            if inverse
                (topology.local_dims_padded[1:3]..., topology.local_dims_padded[4]÷2)
            else
                (topology.local_dims_padded...,)
            end
        end, 0
    end
            
    origin = if struct_name == :Gaugefield
        :(OffsetArrays.Origin(1, topology.bulk_sites[1].I.-halo_width..., 1))
    elseif struct_name == :Spinorfield
        :(OffsetArrays.Origin(1, topology.bulk_sites[1].I.-halo_width...))
    elseif struct_name == :MultiSpinorfield
        :(OffsetArrays.Origin(1, topology.bulk_sites[1].I.-halo_width..., 1))
    elseif struct_name == :Tensorfield
        :(OffsetArrays.Origin(1, topology.bulk_sites[1].I.-halo_width..., 1))
    elseif struct_name == :Colorfield
        :(OffsetArrays.Origin(1, topology.bulk_sites[1].I.-halo_width..., 1))
    elseif struct_name == :Expfield
        :(OffsetArrays.Origin(topology.bulk_sites[1].I.-halo_width..., 1))
    elseif struct_name == :Paulifield
        quote
            ox, oy, oz, ot = topology.bulk_sites[1].I
            if inverse
                ot += topology.local_dims[4] ÷ 2
            end
            OffsetArrays.Origin((ox, oy, oz, ot).-halo_width...,)
        end
    end

    eltype_val = if struct_name == :Gaugefield
        :(SIMD.Vec{N == 18 ? 2 : 4,T})
    elseif struct_name in (:Spinorfield, :MultiSpinorfield)
        :(SIMD.Vec{ND == 1 ? 2 : 4,T})
    elseif struct_name in (:Tensorfield, :Colorfield)
        :(SIMD.Vec{2,T})
    elseif struct_name == :Expfield
        :(ExpiQCoeffs{T})
    elseif struct_name == :Paulifield
        :(PauliMatrix{6,36,T})
    end

    U_construct = :(OffsetArray(bzeros(B(), $eltype_val, $ldims...), $origin))

    sendrecvbuf_dims = if struct_name == :Gaugefield
        :(N == 18 ? 9 : 3, length(border_sites[i][j]), 4)
    elseif struct_name == :Spinorfield
        :(ND == 1 ? 3 : 6, length(border_sites[i][j]))
    elseif struct_name == :MultiSpinorfield
        :(ND == 1 ? 3 : 6, length(border_sites[i][j]), numspinors)
    elseif struct_name == :Tensorfield
        :(9, length(border_sites[i][j]), 6)
    elseif struct_name == :Colorfield
        :(9, length(border_sites[i][j]), 4)
    elseif struct_name == :Expfield
        :(length(border_sites[i][j]), 4)
    elseif struct_name == :Paulifield
        :(length(border_sites[i][j]))
    end

    sendrecvbuf_construct = :(bzeros(B(), $eltype_val, $(sendrecvbuf_dims)))
    return U_construct, sendrecvbuf_construct
end


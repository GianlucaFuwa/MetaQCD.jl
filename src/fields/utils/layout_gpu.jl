function create_gpu_layout(struct_name)
    ldims, tuple_len = if struct_name == :Gaugefield
        :(N == 18 ? 9 : 3, topology.local_dims..., 4), 4
    elseif struct_name == :Spinorfield
        :(ND == 1 ? 3 : 6, topology.local_dims...,), 0
    elseif struct_name == :MultiSpinorfield
        :(ND == 1 ? 3 : 6, topology.local_dims..., numspinors), :numspinors
    elseif struct_name == :Tensorfield
        :(9, topology.local_dims..., 6), 6
    elseif struct_name == :Colorfield
        :(9, topology.local_dims..., 4), 0
    elseif struct_name == :Expfield
        :(topology.local_dims..., 4), 0
    elseif struct_name == :Paulifield
        quote
            if inverse
                (topology.local_dims[1:3]..., topology.local_dims[4]÷2)
            else
                (topology.local_dims...,)
            end
        end, 0
    end
            
    origin = if struct_name == :Gaugefield
        :(OffsetArrays.Origin(1, topology.bulk_sites[1].I..., 1))
    elseif struct_name == :Spinorfield
        :(OffsetArrays.Origin(1, topology.bulk_sites[1].I...,))
    elseif struct_name == :MultiSpinorfield
        :(OffsetArrays.Origin(1, topology.bulk_sites[1].I..., 1))
    elseif struct_name == :Tensorfield
        :(OffsetArrays.Origin(1, topology.bulk_sites[1].I..., 1))
    elseif struct_name == :Colorfield
        :(OffsetArrays.Origin(1, topology.bulk_sites[1].I..., 1))
    elseif struct_name == :Expfield
        :(OffsetArrays.Origin(topology.bulk_sites[1].I..., 1))
    elseif struct_name == :Paulifield
        quote
            ox, oy, oz, ot = topology.bulk_sites[1].I
            if inverse
                ot += topology.local_dims[4] ÷ 2
            end
            OffsetArrays.Origin(ox, oy, oz, ot)
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
    # U_construct = if struct_name in (:Spinorfield, :Paulifield, :Colorfield, :Expfield)
    #     :(OffsetArray(bzeros(B(), $eltype_val, $ldims...), $origin))
    # else
    #     :(ntuple(_ -> OffsetArray(bzeros(B(), $eltype_val, $ldims...), $origin), $tuple_len))
    # end
    
    # Build halo creation (4D for spinors, 5D for others)
    halo_dims, halo_indices = if struct_name == :Gaugefield
        :(N == 18 ? 9 : 3, size(halo_sites[i][j])..., 4),
        :(N == 18 ? 9 : 3, halo_sites[i][j].indices..., 4)
    elseif struct_name == :Spinorfield
        :(ND == 1 ? 3 : 6, size(halo_sites[i][j])...,),
        :(ND == 1 ? 3 : 6, halo_sites[i][j].indices...,)
    elseif struct_name == :MultiSpinorfield
        :(ND == 1 ? 3 : 6, size(halo_sites[i][j])..., numspinors),
        :(ND == 1 ? 3 : 6, halo_sites[i][j].indices..., numspinors)
    elseif struct_name == :Tensorfield
        :(9, size(halo_sites[i][j])..., 6),
        :(9, halo_sites[i][j].indices..., 6)
    elseif struct_name == :Colorfield
        :(9, size(halo_sites[i][j])..., 4),
        :(9, halo_sites[i][j].indices..., 4)
    elseif struct_name == :Expfield
        :(size(halo_sites[i][j])..., 4),
        :(halo_sites[i][j].indices..., 4)
    elseif struct_name == :Paulifield
        :(size(halo_sites[i][j])...,),
        :(halo_sites[i][j].indices...,)
    end

    halo_construct = :(OffsetArray(bzeros(B(), $eltype_val, $(halo_dims.args...)), $(halo_indices.args...)))
    # halo_construct = if struct_name in (:Spinorfield, :Paulifield)
    #     :(OffsetArray(bzeros(B(), $eltype_val, $(halo_dims.args...)), $(halo_indices.args...)))
    # else
    #     quote
    #         ntuple($(tuple_len)) do i
    #             OffsetArray(bzeros(B(), $eltype_val, $(halo_dims.args...)), $(halo_indices.args...))
    #         end
    #     end
    # end

    sendbuf_dims = if struct_name == :Gaugefield
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

    sendbuf_construct = :(bzeros(B(), $eltype_val, $(sendbuf_dims)))
    # sendbuf_construct = if struct_name in (:Spinorfield, :Paulifield, :Colorfield, :Expfield)
    #     :(bzeros(B(), $eltype_val, $(sendbuf_dims)))
    # else
    #     quote
    #         ntuple($(tuple_len)) do i
    #             bzeros(B(), $eltype_val, $(sendbuf_dims))
    #         end
    #     end
    # end

    return U_construct, halo_construct, sendbuf_construct
end


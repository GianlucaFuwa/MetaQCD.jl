function create_gpu_layout(struct_name)
    ldims, tuple_len = if struct_name == :Gaugefield
        :(topology.local_dims..., N == 18 ? 9 : 3, 4), 4
    elseif struct_name == :Spinorfield
        :(topology.local_dims..., ND == 1 ? 3 : 6), 0
    elseif struct_name == :MultiSpinorfield
        :(topology.local_dims..., ND == 1 ? 3 : 6, numspinors), :numspinors
    elseif struct_name == :Tensorfield
        :(topology.local_dims..., 9, 6), 6
    elseif struct_name == :Colorfield
        :(topology.local_dims..., 9, 4), 0
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
        :(OffsetArrays.Origin(topology.bulk_sites[1].I..., 1, 1))
    elseif struct_name == :Spinorfield
        :(OffsetArrays.Origin(topology.bulk_sites[1].I..., 1))
    elseif struct_name == :MultiSpinorfield
        :(OffsetArrays.Origin(topology.bulk_sites[1].I..., 1, 1))
    elseif struct_name == :Tensorfield
        :(OffsetArrays.Origin(topology.bulk_sites[1].I..., 1, 1))
    elseif struct_name == :Colorfield
        :(OffsetArrays.Origin(topology.bulk_sites[1].I..., 1, 1))
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
        :(size(halo_sites[i][j])..., N == 18 ? 9 : 3, 4),
        :(halo_sites[i][j].indices..., 1:(N == 18 ? 9 : 3), 1:4)
    elseif struct_name == :Spinorfield
        :(size(halo_sites[i][j])..., ND == 1 ? 3 : 6),
        :(halo_sites[i][j].indices..., 1:(ND == 1 ? 3 : 6))
    elseif struct_name == :MultiSpinorfield
        :(size(halo_sites[i][j])..., ND == 1 ? 3 : 6, numspinors),
        :(halo_sites[i][j].indices..., 1:(ND == 1 ? 3 : 6), 1:numspinors)
    elseif struct_name == :Tensorfield
        :(size(halo_sites[i][j])..., 9, 6),
        :(halo_sites[i][j].indices..., 1:9, 1:6)
    elseif struct_name == :Colorfield
        :(size(halo_sites[i][j])..., 9, 4),
        :( halo_sites[i][j].indices..., 1:9, 1:4)
    elseif struct_name == :Expfield
        :(size(halo_sites[i][j])..., 4),
        :(halo_sites[i][j].indices..., 1:4)
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
        :(length(border_sites[i][j]), N == 18 ? 9 : 3, 4)
    elseif struct_name == :Spinorfield
        :(length(border_sites[i][j]), ND == 1 ? 3 : 6)
    elseif struct_name == :MultiSpinorfield
        :(length(border_sites[i][j]), ND == 1 ? 3 : 6, numspinors)
    elseif struct_name == :Tensorfield
        :(length(border_sites[i][j]), 9, 6)
    elseif struct_name == :Colorfield
        :(length(border_sites[i][j]), 9, 4)
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


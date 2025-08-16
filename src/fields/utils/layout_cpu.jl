function create_cpu_layout(struct_name)
    ldims, inner_len = if struct_name == :Gaugefield
        :(4, topology.local_dims...,), 4
    elseif struct_name == :Spinorfield
        :(topology.local_dims,), 0
    elseif struct_name == :MultiSpinorfield
        :(numspinors, topology.local_dims...,), :numspinors
    elseif struct_name == :Tensorfield
        :(6, topology.local_dims...,), 6
    elseif struct_name == :Colorfield
        :(4, topology.local_dims...,), 6
    elseif struct_name == :Expfield
        :(4, topology.local_dims...,), 6
    elseif struct_name == :Paulifield
        quote
            if inverse
                (topology.local_dims[1:3]..., topology.local_dims[4]÷2)
            else
                topology.local_dims
            end
        end, 0
    end
            
    origin = if struct_name == :Spinorfield
        :(OffsetArrays.Origin(topology.bulk_sites[1]))
    elseif struct_name == :Paulifield
        quote
            ox, oy, oz, ot = topology.bulk_sites[1].I
            if inverse
                ot += topology.local_dims[4] ÷ 2
            end
            OffsetArrays.Origin(ox, oy, oz, ot)
        end
    else
        :(OffsetArrays.Origin(1, (topology.bulk_sites[1].I)...))
    end

    eltype_val = if struct_name in (:Spinorfield, :MultiSpinorfield)
        :(eltype($(struct_name), T, Val(ND)))
    else
        :(eltype($(struct_name), T))
    end

    U_construct = :(OffsetArray(zeros($eltype_val, $ldims...), $origin))
    
    # Build halo creation (4D for spinors, 5D for others)
    halo_dims, halo_indices = if struct_name in (:Spinorfield, :Paulifield)
        :(size(halo_sites[i][j])...), :(halo_sites[i][j].indices...,)
    else
        :($inner_len, size(halo_sites[i][j])...),
        :(1:$inner_len, halo_sites[i][j].indices...)
    end

    halo_construct = :(OffsetArray(zeros($eltype_val, $(halo_dims.args...)), $(halo_indices.args...)))

    sendbuf_dims = if struct_name in (:Spinorfield, :Paulifield)
        :(length(border_sites[i][j]))
    else
        :($inner_len, length(border_sites[i][j])...)
    end

    sendbuf_construct = :(zeros($eltype_val, $(sendbuf_dims)))
    return U_construct, halo_construct, sendbuf_construct
end


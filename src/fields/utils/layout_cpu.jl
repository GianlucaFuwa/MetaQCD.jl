function create_cpu_layout(struct_name)
    ldims, inner_len = if struct_name == :Gaugefield
        :(4, topology.local_dims.+2halo_width...,), 4
    elseif struct_name == :Spinorfield
        :(topology.local_dims.+2halo_width...,), 1
    elseif struct_name == :MultiSpinorfield
        :(numspinors, topology.local_dims.+2halo_width...,), :numspinors
    elseif struct_name == :Tensorfield
        :(6, topology.local_dims.+2halo_width...,), 6
    elseif struct_name == :Colorfield
        :(4, topology.local_dims.+2halo_width...,), 6
    elseif struct_name == :Expfield
        :(4, topology.local_dims.+2halo_width...,), 6
    elseif struct_name == :Paulifield
        quote
            if inverse
                (topology.local_dims[1:3]..., topology.local_dims[4]÷2)
            else
                topology.local_dims
            end
        end, 1
    end
            
    origin = if struct_name == :Spinorfield
        :(OffsetArrays.Origin(topology.bulk_sites[1].I.-halo_width...,))
    elseif struct_name == :Paulifield
        quote
            ox, oy, oz, ot = topology.bulk_sites[1].I
            if inverse
                ot += topology.local_dims[4] ÷ 2
            end
            OffsetArrays.Origin((ox, oy, oz, ot).-halo_width...,)
        end
    else
        :(OffsetArrays.Origin(1, (topology.bulk_sites[1].I.-halo_width)...))
    end

    eltype_val = if struct_name in (:Spinorfield, :MultiSpinorfield)
        :(eltype($(struct_name), T, Val(ND)))
    else
        :(eltype($(struct_name), T))
    end

    U_construct = :(OffsetArray(zeros($eltype_val, $ldims...), $origin))

    sendrecvbuf_dims = :($inner_len * length(border_sites[i][j])...,)

    sendrecvbuf_construct = :(zeros($eltype_val, $(sendrecvbuf_dims)))
    return U_construct, sendrecvbuf_construct
end


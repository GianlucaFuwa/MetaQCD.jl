"""
    @field_constructor StructName [extra_types=()] [extra_fields=()]

Generate a field constructor with standard initialization.
All structs have B, T as first two type parameters.

# Examples:
```julia
@field_constructor Colorfield                                   # -> Colorfield{B,T}(NX, NY, NZ, NT; numprocs_cart, halo_width)
@field_constructor Gaugefield extra_types=GA extra_fields=β     # -> Gaugefield{B,T,GA}(NX, NY, NZ, NT, β; numprocs_cart, halo_width)
...
```
"""
macro field_constructor(struct_name, kwargs...)
    @assert length(kwargs) <= 2
    kwdict = Dict{Symbol,Any}(:extra_types => (), :extra_args => ())
    for el in kwargs
        if Meta.isexpr(el, :(=))
            kwdict[el.args[1]] = if el.args[2] isa Symbol
                (el.args[2],)
            else
                (el.args[2].args...,)
            end
        end
    end
    extra_types = kwdict[:extra_types]
    extra_args = kwdict[:extra_args]

    is_spinorfield = struct_name == :Spinorfield
    ldims_q, inner_len = if is_spinorfield
        :(topology.local_dims,), 0
    elseif struct_name == :Paulifield
        pauli_ldims = quote
            if inverse
                (topology.local_dims[1:3]..., topology.local_dims[4]÷2)
            else
                topology.local_dims
            end
        end
        pauli_ldims, 0
    elseif struct_name == :Tensorfield
        :(6, topology.local_dims...), 6
    elseif struct_name == :MultiSpinorfield
        :(numspinors, topology.local_dims...), :numspinors
    else
        :(4, topology.local_dims...), 4
    end
            
    origin_q = if is_spinorfield
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
    
    # Build halo creation (4D for spinors, 5D for others)
    halo_dims, halo_indices = if is_spinorfield || struct_name == :Paulifield
        :(size(halo_sites[i][j])...), :(halo_sites[i][j].indices...,)
    else
        :($inner_len, size(halo_sites[i][j])...),
        :(1:$inner_len, halo_sites[i][j].indices...)
    end

    sendbuf_dims = if is_spinorfield || struct_name == :Paulifield
        :(length(border_sites[i][j]))
    else
        :($inner_len, length(border_sites[i][j])...)
    end

    halo_check = if struct_name == :Gaugefield
        quote
            if numprocs > 1 && !no_halo
                @assert halo_width >= stencil_size(GA) """
                halo_width must be >= 2 when using improved gauge actions
                """
            end
        end
    else
        Expr(:block)
    end

    eltype_q = if struct_name in (:Spinorfield, :MultiSpinorfield)
        :(eltype($(struct_name), T, Val(ND)))
    else
        :(eltype($(struct_name), T))
    end

    # additional struct fields
    extra_fields = if struct_name == :Gaugefield
        :(β::Float64)
    elseif struct_name == :MultiSpinorfield
        :(numspinors::Int64)
    elseif struct_name == :Paulifield
        quote
            csw::Float64
            inverse::Bool
        end
    else
        Expr(:block)
    end
    
    # Build final constructor call arguments
    base_args = [:NX, :NY, :NZ, :NT, extra_args...]
    final_args = [:U, :halos, :sendbuf, :topology, extra_args...]
    base_types, additional_ex = if struct_name == :Paulifield
        [:B, :T], :(C = csw != 0)
    else
        [:B, :T, extra_types...], :()
    end

    # Generate the complete constructor
    constructor_expr = quote
        struct $(struct_name){B,T,M,$(extra_types...),AT,HT,BT,TT,HV} <: AbstractField{B,T,M}
            U::AT
            halos::HT
            sendbuf::BT
            topology::TT
            $(extra_fields)
            halo_valid::HV
            function $(struct_name){B,T,M,$(extra_types...)}(
                U::AT, halos::HT, sendbuf::BT, topology::TT, $(extra_args...), halo_valid::HV
            ) where {B,T,M,$(extra_types...),AT,HT,BT,TT,HV}
                check_types(B, T, U, halos, sendbuf)
                return new{B,T,M,$(extra_types...),AT,HT,BT,TT,HV}(
                    U, halos, sendbuf, topology, $(extra_args...), halo_valid
                )
            end
        end

        function $(struct_name){$(base_types...)}(
            $(base_args...);
            numprocs_cart=(1, 1, 1, 1), halo_width=0, no_halo=false, halo_valid=Ref(false)
        ) where {$(base_types...)}
            numprocs = prod(numprocs_cart)
            M = numprocs > 1 && !no_halo

            if !M
                halo_width = 0
            end

            $halo_check
            topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))

            # Create U array
            eltype_val = $eltype_q
            origin = $origin_q
            ldims = $ldims_q
            U = OffsetArray(bzeros(B(), eltype_val, ldims...), origin)
            # Create halos and sendbuf
            halo_sites = topology.halo_sites
            border_sites = topology.border_sites

            halos = if M
                tuple([
                    OffsetArray(
                        bzeros(B(), eltype_val, $(halo_dims.args...)),
                        $(halo_indices.args...)
                    )
                    for i in 1:4 for j in 1:2
                ]...)
            else
                nothing
            end

            sendbuf = if M
                tuple(
                    [bzeros(B(), eltype_val, $(sendbuf_dims)) for i in 1:4 for j in 1:2]
                    ...)
            else
                nothing
            end
            # Return constructed object
            $additional_ex
            return $(struct_name){B,T,M,$(extra_types...)}($(final_args...), halo_valid)
        end
    end

    return esc(constructor_expr)
end

# XXX: Legacy
# function Colorfield{B,T}(NX, NY, NZ, NT) where {B,T}
#     U = KA.zeros(B(), SU{3,9,T}, 4, NX, NY, NZ, NT)
#     halos = nothing
#     numprocs_cart = (1, 1, 1, 1)
#     halo_width = 0
#     topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
#     return Colorfield{B,T,false}(U, halos, topology)
# end
#
# function Colorfield{B,T}(NX, NY, NZ, NT, numprocs_cart, halo_width; nohalo=false) where {B,T}
#     if prod(numprocs_cart) == 1
#         return Colorfield{B,T}(NX, NY, NZ, NT)
#     end
#
#     topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
#     ldims = nohalo ? topology.local_dims : topology.local_dims
#     eltype = SMatrix{3,3,Complex{T},9}
#
#     origin = OffsetArrays.Origin((1, (topology.bulk_sites[1].I .- halo_width)...)...)
#     U = OffsetArray(KA.zeros(B(), eltype, 4, ldims...), origin)
#     halo_sites = topology.halo_sites
#     halos = [
#         OffsetArray(
#             KA.zeros(B(), eltype, 4, size(halo_sites[i][j])...),
#             1:4, halo_sites[i][j].indices...
#         )
#         for i in 1:4 for j in 1:2
#     ]
#     return Colorfield{B,T,true}(U, halos, topology)
# end

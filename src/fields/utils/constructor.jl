"""
    @field_constructor StructName [extra_types=()] [extra_fields=()]

Generate a field constructor with standard initialization.
All structs have B, T as first two type parameters.

# Examples:
```julia
@field_constructor Colorfield                                   # -> Colorfield{B,T}(NX, NY, NZ, NT; numprocs_cart, halo_width)
@field_constructor Gaugefield extra_types=GA,N extra_fields=β   # -> Gaugefield{B,T,GA,N}(NX, NY, NZ, NT, β; numprocs_cart, halo_width)
...
```
"""
macro field_constructor(struct_name, kwargs...)
    extra_types, extra_args = extract_constructor_extras(kwargs...)
    # Build final constructor call arguments
    base_args = [:NX, :NY, :NZ, :NT, extra_args...]
    final_args = [:U, :sendbuf, :recvbuf, :topology, extra_args...]
    additional_ex = struct_name == :Paulifield ? :(C = csw != 0) : :()

    struct_def = quote
        struct $(struct_name){B,T,M,$(extra_types...),AT,BT,TT,HV} <: AbstractField{B,T,M}
            U::AT
            sendbuf::BT
            recvbuf::BT
            topology::TT
            $(extra_fields(struct_name))
            halo_valid::HV
            function $(struct_name){B,T,M,$(extra_types...)}(
                U::AT, sendbuf::BT, recvbuf::BT, topology::TT, $(extra_args...), halo_valid::HV
            ) where {B,T,M,$(extra_types...),AT,BT,TT,HV}
                check_types(B, T, U, sendbuf, recvbuf)
                return new{B,T,M,$(extra_types...),AT,BT,TT,HV}(
                    U, sendbuf, recvbuf, topology, $(extra_args...), halo_valid
                )
            end
        end
    end

    constructor = []

    for base_types in [[:CPU, :T, extra_types...], [:B, :T, extra_types...]]
        struct_name == :Paulifield && (base_types = [base_types[1], :T])
        var_types, (U_construct, sendrecvbuf_construct) = if base_types[1] == :CPU
            base_types[2:end], create_cpu_layout(struct_name)
        else
            base_types, create_gpu_layout(struct_name)
        end

        push!(
            constructor,
            quote
                function $(struct_name){$(base_types...)}(
                    $(base_args...);
                    numprocs_cart=(1, 1, 1, 1), halo_width=0, no_halo=false, halo_valid=Ref(false)
                ) where {$(var_types...)}
                    numprocs = prod(numprocs_cart)
                    M = numprocs > 1 && !no_halo
                    M || (halo_width = 0)

                    $(halo_check(struct_name)) # if Gaugefield, check that halo is wide enough for gauge action
                    topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, NT))
                    eff_halo_width = halo_width .* (numprocs_cart.>1)

                    # Create U array
                    mpi_assign_device!($(base_types[1])(), mpi_myrank())

                    U = $U_construct
                    # Create send- and recvbuf
                    halo_sites = topology.halo_sites
                    border_sites = topology.border_sites
                    numpart = 4

                    sendbuf = if M
                        tuple([$(sendrecvbuf_construct) for i in 1:numpart for j in 1:2]...)
                    else
                        nothing
                    end

                    recvbuf = if M
                        tuple([$(sendrecvbuf_construct) for i in 1:numpart for j in 1:2]...)
                    else
                        nothing
                    end
                    # Return constructed object
                    $additional_ex
                    return $(struct_name){$(base_types[1]),T,M,$(extra_types...)}(
                        $(final_args...), halo_valid
                    )
                end
            end
        )
    end

    # Generate the complete constructor
    constructor_expr = quote
        $struct_def
        $(constructor[1])
        $(constructor[2])
    end

    return esc(constructor_expr)
end

function extract_constructor_extras(kwargs...)
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
    return extra_types, extra_args
end

function halo_check(struct_name)
    return if struct_name == :Gaugefield
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
end

function extra_fields(struct_name)
    # additional struct fields
    return if struct_name == :Gaugefield
        :(β::Float64)
    elseif struct_name == :MultiSpinorfield
        :(numspinors::Int64)
    elseif struct_name == :Paulifield
        quote
            csw::Float64
            inverse::Bool
        end
    else
        Expr(:block) # empty block
    end
end

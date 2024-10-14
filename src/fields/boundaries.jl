# INFO: The methods in this file are only meant to be used with Spinorfields and
# only in the time direction for now 

# To determine whether the local partition of a distributed field includes a boundary
abstract type AbstractBoundaryMode end

struct NoBoundary <: AbstractBoundaryMode end
struct NegBoundary <: AbstractBoundaryMode end
struct PosBoundary <: AbstractBoundaryMode end
struct BothBoundaries <: AbstractBoundaryMode end

abstract type AbstractBoundaryCondition end

struct PeriodicBC <: AbstractBoundaryCondition end
struct AntiPeriodicBC{T<:AbstractBoundaryMode} <: AbstractBoundaryCondition end

const BOUNDARY_CONDITIONS = Dict(
    "periodic" => PeriodicBC,
    "antiperiodic" => AntiPeriodicBC,
)

@inline get_bctype(str::String) = BOUNDARY_CONDITIONS[str]

@inline create_bc(str, topology::FieldTopology) = create_bc(get_bctype(str), topology)
@inline create_bc(::Type{PeriodicBC}, ::FieldTopology) = PeriodicBC()

@inline function create_bc(::Type{AntiPeriodicBC}, topology::FieldTopology)
    local_ranges = topology.local_ranges
    global_dims = topology.global_dims
    min_it = local_ranges[4][1]
    max_it = local_ranges[4][end]

    bc_mode = if min_it == 1
        if max_it == global_dims[4]
            BothBoundaries
        else
            NegBoundary
        end
    else
        if max_it == global_dims[4]
            PosBoundary
        else
            NoBoundary
        end
    end

    return AntiPeriodicBC{bc_mode}()
end

"""
    apply_bc(el, bc::AbstractBoundaryCondition, site, ::Val{dir}, NT::Int)

Apply the boundary condition `bc` on element `el`, which is assumed to lie at a site
adjacent to `site` within an `AbstractField`.
`dir` ∈ {-1, 1} specifies whether `el` is a negative or positive neighbor and `NT`
is the maximum time extent.
"""
@inline apply_bc(::Any, ::PeriodicBC, ::SiteCoords, ::Val{dir}, ::Int64) where {dir} = el   

@generated function apply_bc(el, bc::AntiPeriodicBC{T}, site, ::Val{dir}, NT) where {T,dir}
    q = quote
        $(Expr(:meta, :inline))
        it = site[4]
    end

    if T === NoBoundary
        push!(q.args, :(return el))
    else
        if dir == 1
            if T === NegBoundary
                push!(q.args, :(return el))
            else
                push!(q.args, :(return (it == NT ? -1 : 1) * el))
            end
        elseif dir == -1
            if T === PosBoundary
                push!(q.args, :(return el))
            else
                push!(q.args, :(return (it == 1 ? -1 : 1) * el))
            end
        else
            throw(ArgumentError("dir must be either Val(-1) or Val(1) in apply_bc"))
        end
    end

    return q
end

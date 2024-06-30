"""
    Gaugefield{BACKEND,T,GA}(NX, NY, NZ, NT, β)
    Gaugefield(U::Gaugefield)

Creates a Gaugefield on `BACKEND`, i.e. an array of link-variables (SU3 matrices with `T`
precision) of size `4 × NX × NY × NZ × NT` with coupling parameter `β` and gauge action `GA`
or a zero-initialized copy of `U`
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
# Supported gauge actions
`WilsonGaugeAction` \\
`SymanzikTreeGaugeAction` (Lüscher-Weisz) \\
`IwasakiGaugeAction` \\
`DBW2GaugeAction`
"""
struct Gaugefield{BACKEND,T,A,GA} <: Abstractfield{BACKEND,T,A}
    U::A # Actual field storing the gauge variables
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors

    β::Float64 # Seems weird to have it here, but I couldnt be bothered passing it as an argument everywhere
    Sg::Base.RefValue{Float64} # Current Gauge action, used to safe work
    CV::Base.RefValue{Float64} # Current collective variable, used to safe work
    function Gaugefield{BACKEND,T,GA}(NX, NY, NZ, NT, β) where {BACKEND,T,GA}
        U = KA.zeros(BACKEND(), SU{3,9,T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        Sg = Base.RefValue{Float64}(0.0)
        CV = Base.RefValue{Float64}(0.0)
        return new{BACKEND,T,typeof(U),GA}(U, NX, NY, NZ, NT, NV, 3, β, Sg, CV)
    end
end

function Gaugefield(parameters)
    NX, NY, NZ, NT = parameters.L
    β = parameters.beta
    GA = GAUGE_ACTION[parameters.gauge_action]
    T = Utils.FLOAT_TYPE[parameters.float_type]
    B = BACKEND[parameters.backend]
    U = Gaugefield{B,T,GA}(NX, NY, NZ, NT, β)

    initial = parameters.initial
    if initial == "cold"
        identity_gauges!(U)
    elseif initial == "hot"
        random_gauges!(U)
    else
        error("intial condition \"$(initial)\" not supported, only \"cold\" or \"hot\"")
    end

    return U
end

function Gaugefield(u::Gaugefield{BACKEND,T,A,GA}) where {BACKEND,T,A,GA}
    return Gaugefield{BACKEND,T,GA}(u.NX, u.NY, u.NZ, u.NT, u.β)
end

"""
    Colorfield{BACKEND,T}(NX, NY, NZ, NT)
    Colorfield(u::Abstractfield)

Creates a Colorfield on `BACKEND`, i.e. an array of 3-by-3 `T`-precision matrices of
size `4 × NX × NY × NZ × NT` or a zero-initialized Colorfield of the same size as `u`
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct Colorfield{BACKEND,T,A} <: Abstractfield{BACKEND,T,A}
    U::A
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NV::Int64
    NC::Int64
    function Colorfield{BACKEND,T}(NX, NY, NZ, NT) where {BACKEND,T}
        U = KA.zeros(BACKEND(), SU{3,9,T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        NC = 3
        return new{BACKEND,T,typeof(U)}(U, NX, NY, NZ, NT, NV, NC)
    end
end

function Colorfield(u::Abstractfield{BACKEND,T,A}) where {BACKEND,T,A}
    return Colorfield{BACKEND,T}(u.NX, u.NY, u.NZ, u.NT)
end

"""
    Expfield{BACKEND,T}(NX, NY, NZ, NT)
    Expfield(u::Abstractfield)

Creates a Expfield on `BACKEND`, i.e. an array of `T`-precison `exp_iQ_su3` objects of
size `4 × NX × NY × NZ × NT` or of the same size as `u`. The objects hold the `Q`-matrices
and all the exponential parameters needed for stout-force recursion
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
struct Expfield{BACKEND,T,A} <: Abstractfield{BACKEND,T,A}
    U::A # TODO: Add support for arbitrary NC
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NV::Int64
    function Expfield{BACKEND,T}(NX, NY, NZ, NT) where {BACKEND,T}
        U = KA.zeros(BACKEND(), exp_iQ_su3{T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        return new{BACKEND,T,typeof(U)}(U, NX, NY, NZ, NT, NV)
    end
end

function Expfield(u::Abstractfield{BACKEND,T,A}) where {BACKEND,T,A}
    return Expfield{BACKEND,T}(u.NX, u.NY, u.NZ, u.NT)
end

# XXX: might not be needed
Base.eltype(u::Abstractfield) = eltype(u.U)
Base.elsize(u::Abstractfield) = Base.elsize(u.U)
Base.parent(u::Abstractfield) = u.U
Base.pointer(u::Abstractfield) = pointer(u.U)
Base.strides(u::Abstractfield) = strides(u.U)
# This converts u to a PtrArray pointing to the entries of u.U, meaning that we cant
# access any of the fields of u within the @batch loop
@inline object_and_preserve(u::Abstractfield) = object_and_preserve(u.U)
float_type(::AbstractArray{SMatrix{3,3,Complex{T},9},5}) where {T} = T
float_type(::Abstractfield{BACKEND,T}) where {BACKEND,T} = T
KA.get_backend(u::Abstractfield) = get_backend(u.U)

# So we don't print the entire array in the REPL...
function Base.show(io::IO, ::MIME"text/plain", u::T) where {T<:Abstractfield}
    print(io, "$(typeof(u))", "(;")
    for fieldname in fieldnames(T)
        fieldname ∈ (:U, :NV) && continue

        if fieldname ∈ (:Sf, :Sg, :CV)
            print(io, " ", fieldname, " = ", getfield(u, fieldname)[], ",")
        else
            print(io, " ", fieldname, " = ", getfield(u, fieldname), ",")
        end
    end
    print(io, ")")
    return nothing
end

function Base.show(io::IO, u::T) where {T<:Abstractfield}
    print(io, "$(typeof(u))", "(;")
    for fieldname in fieldnames(T)
        fieldname ∈ (:U, :NV) && continue

        if fieldname ∈ (:Sf, :Sg, :CV)
            print(io, " ", fieldname, " = ", getfield(u, fieldname)[], ",")
        else
            print(io, " ", fieldname, " = ", getfield(u, fieldname), ",")
        end
    end
    print(io, ")")
    return nothing
end

# define dims() function twice --- once for generic arrays, such that GPUs and @batch
# can use it, and once for Abstractfields for any other case
@inline dims(u) = NTuple{4,Int64}((size(u, 2), size(u, 3), size(u, 4), size(u, 5)))
@inline dims(u::Abstractfield) = NTuple{4,Int64}((u.NX, u.NY, u.NZ, u.NT))
@inline volume(u) = prod(dims(u))
@inline volume(u::Abstractfield) = u.NV
Base.ndims(u::Abstractfield) = 4
Base.size(u::Abstractfield) = NTuple{5,Int64}((4, u.NX, u.NY, u.NZ, u.NT))

"""
    check_dims(x1, rest...)

Check if all fields have the same dimensions. Throw an `AssertionError` otherwise.
"""
@generated function check_dims(x1, rest::Vararg{Any,N}) where {N}
    q_inner = Expr(:comparison, :(dims(x1)))
    for i in 1:N
        push!(q_inner.args, :(==))
        push!(q_inner.args, :(dims(rest[$i])))
    end
    q = Expr(:macrocall, Symbol("@assert"), :(), q_inner)
    return q
end

Base.eachindex(u::Abstractfield) = CartesianIndices((u.NX, u.NY, u.NZ, u.NT))
Base.eachindex(::IndexLinear, u::Abstractfield) = Base.OneTo(u.NV)
function Base.eachindex(even::Bool, u::Abstractfield)
    NX, NY, NZ, NT = dims(u)
    @assert iseven(NT)
    last_range = even ? (1:div(NT, 2)) : (div(NT, 2)+1:NT)
    return CartesianIndices((NX, NY, NZ, last_range))
end
Base.length(u::Abstractfield) = u.NV

gauge_action(::Gaugefield{B,T,A,GA}) where {B,T,A,GA} = GA

# overload get and set for the Abstractfields structs, so we dont have to do u.U[μ,x,y,z,t]
Base.@propagate_inbounds Base.getindex(u::Abstractfield, μ, x, y, z, t) = u.U[μ, x, y, z, t]
Base.@propagate_inbounds Base.getindex(u::Abstractfield, μ, site::SiteCoords) = u.U[μ, site]
Base.@propagate_inbounds Base.setindex!(u::Abstractfield, v, μ, x, y, z, t) =
    setindex!(u.U, v, μ, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(u::Abstractfield, v, μ, site::SiteCoords) =
    setindex!(u.U, v, μ, site)

# overload getproperty and setproperty! for convenience
function Base.getproperty(u::Gaugefield, p::Symbol)
    if p == :Sg
        return getfield(u, :Sg)[]
    elseif p == :CV
        return getfield(u, :CV)[]
    else
        return getfield(u, p)
    end
end

function Base.setproperty!(u::Gaugefield, p::Symbol, val)
    if p == :Sg
        getfield(u, :Sg)[] = val
    elseif p == :CV
        getfield(u, :CV)[] = val
    else
        setproperty!(u, p, val)
    end

    return nothing
end

Base.similar(u::Gaugefield) = Gaugefield(u)
Base.similar(u::Colorfield) = Colorfield(u)
Base.similar(u::Expfield) = Expfield(u)

"""
	to_backend(Bout, u::Abstractfield{Bin,T})

Ports the Abstractfield u to the backend Bout, maintaining all elements
# Supported backends
`CPU` \\
`CUDABackend` \\
`ROCBackend`
"""
function to_backend(::Type{Bout}, u::Abstractfield{Bin,T}) where {Bout,Bin,T}
    Bout === Bin && return u # no need to do anything if the backends are the same
    A = array_type(Bout)
    sizeU = dims(u)
    Uout = A(u.U)

    if u isa Gaugefield
        GA = gauge_action(u)
        Sg = Base.RefValue{Float64}(u.Sg)
        CV = Base.RefValue{Float64}(u.CV)
        return Gaugefield{Bout,T,typeof(Uout),GA}(Uout, sizeU..., u.NV, 3, u.β, Sg, CV)
    elseif u isa Expfield
        return Expfield{Bout,T,typeof(Uout)}(Uout, sizeU..., u.NV, 3)
    elseif u isa Colorfield
        return Colorfield{Bout,T,typeof(Uout)}(Uout, sizeU..., u.NV, 3)
    elseif u isa Tensorfield
        return Tensorfield{Bout,T,typeof(Uout)}(Uout, sizeU..., u.NV, 3)
    else
        throw(ArgumentError("Unsupported field type"))
    end
end

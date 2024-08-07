"""
    Gaugefield{BACKEND,T,GA}(NX, NY, NZ, NT, β)
    Gaugefield(U::Gaugefield)
    Gaugefield(parameters::ParameterSet)

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
    function Gaugefield{BACKEND,T,A,GA}(U::A, NX, NY, NZ, NT, NV, NC, β, Sg, CV) where {BACKEND,T,A,GA}
        return new{BACKEND,T,A,GA}(U, NX, NY, NZ, NT, NV, NC, β, Sg, CV)
    end

    function Gaugefield{BACKEND,T,GA}(NX, NY, NZ, NT, β) where {BACKEND,T,GA}
        U = KA.zeros(BACKEND(), SU{3,9,T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        Sg = Base.RefValue{Float64}(0.0)
        CV = Base.RefValue{Float64}(0.0)
        return new{BACKEND,T,typeof(U),GA}(U, NX, NY, NZ, NT, NV, 3, β, Sg, CV)
    end

    function Gaugefield(u::Gaugefield{BACKEND,T,A,GA}) where {BACKEND,T,A,GA}
        return Gaugefield{BACKEND,T,GA}(u.NX, u.NY, u.NZ, u.NT, u.β)
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
    function Colorfield{BACKEND,T,A}(U::A, NX, NY, NZ, NT, NV, NC) where {BACKEND,T,A}
        return new{BACKEND,T,A}(U, NX, NY, NZ, NT, NV, NC)
    end

    function Colorfield{BACKEND,T}(NX, NY, NZ, NT) where {BACKEND,T}
        U = KA.zeros(BACKEND(), SU{3,9,T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        NC = 3
        return new{BACKEND,T,typeof(U)}(U, NX, NY, NZ, NT, NV, NC)
    end

    function Colorfield(u::Abstractfield{BACKEND,T,A}) where {BACKEND,T,A}
        return Colorfield{BACKEND,T}(u.NX, u.NY, u.NZ, u.NT)
    end
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
    U::A
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NV::Int64
    function Expfield{BACKEND,T,A}(U::A, NX, NY, NZ, NT, NV, NC) where {BACKEND,T,A}
        return new{BACKEND,T,A}(U, NX, NY, NZ, NT, NV, NC)
    end

    function Expfield{BACKEND,T}(NX, NY, NZ, NT) where {BACKEND,T}
        U = KA.zeros(BACKEND(), exp_iQ_su3{T}, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        return new{BACKEND,T,typeof(U)}(U, NX, NY, NZ, NT, NV)
    end

    function Expfield(u::Abstractfield{BACKEND,T,A}) where {BACKEND,T,A}
        return Expfield{BACKEND,T}(u.NX, u.NY, u.NZ, u.NT)
    end
end

gauge_action(::Gaugefield{B,T,A,GA}) where {B,T,A,GA} = GA

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

abstract type AbstractFieldstrength end

struct Plaquette <: AbstractFieldstrength end
struct Clover <: AbstractFieldstrength end
struct Improved <: AbstractFieldstrength end

struct Tensorfield{BACKEND,T,A<:AbstractArray{SU{3,9,T},6}} <: Abstractfield{BACKEND,T,A}
    U::A
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NV::Int64
    NC::Int64
    function Tensorfield(NX, NY, NZ, NT; BACKEND=CPU, T=Float64)
        @assert BACKEND ∈ SUPPORTED_BACKENDS "Only CPU, CUDABackend or ROCBackend supported!"
        # TODO: Reduce size of Tensorfield by using symmetry of the fieldstrength tensor
        U = KA.zeros(BACKEND(), SMatrix{3,3,Complex{T},9}, 4, 4, NX, NY, NZ, NT)
        NV = NX * NY * NZ * NT
        NC = 3
        return Tensorfield{BACKEND,T,typeof(U)}(U, NX, NY, NZ, NT, NV, NC)
    end
end

function Tensorfield(u::Abstractfield{BACKEND,T,A}) where {BACKEND,T,A}
    NX, NY, NZ, NT = dims(u)
    U = KA.zeros(BACKEND(), SU{3,9,T}, 4, 4, NX, NY, NZ, NT)
    return Tensorfield{BACKEND,T,typeof(U)}(U, NX, NY, NZ, NT, u.NV, u.NC)
end

# overload get and set for the Tensorfields, so we dont have to do u.U[μ,ν,x,y,z,t]
Base.@propagate_inbounds Base.getindex(u::Abstractfield, μ, ν, x, y, z, t) =
    u.U[μ, ν, x, y, z, t]
Base.@propagate_inbounds Base.getindex(u::Abstractfield, μ, ν, site::SiteCoords) =
    u.U[μ, ν, site]
Base.@propagate_inbounds Base.setindex!(u::Abstractfield, v, μ, ν, x, y, z, t) =
    setindex!(u.U, v, μ, ν, x, y, z, t)
Base.@propagate_inbounds Base.setindex!(u::Abstractfield, v, μ, ν, site::SiteCoords) =
    setindex!(u.U, v, μ, ν, site)

function fieldstrength_eachsite!(F::Tensorfield, U, kind_of_fs::String)
    if kind_of_fs == "plaquette"
        fieldstrength_eachsite!(Plaquette(), F, U)
    elseif kind_of_fs == "clover"
        fieldstrength_eachsite!(Clover(), F, U)
    else
        error("kind of fieldstrength \"$(kind_of_fs)\" not supported")
    end

    return nothing
end

function fieldstrength_eachsite!(::Plaquette, F::Tensorfield{CPU}, U::Gaugefield{CPU})
    @assert dims(F) == dims(U)

    @batch for site in eachindex(U)
        C12 = plaquette(U, 1, 2, site)
        F[1, 2, site] = im * traceless_antihermitian(C12)
        C13 = plaquette(U, 1, 3, site)
        F[1, 3, site] = im * traceless_antihermitian(C13)
        C14 = plaquette(U, 1, 4, site)
        F[1, 4, site] = im * traceless_antihermitian(C14)
        C23 = plaquette(U, 2, 3, site)
        F[2, 3, site] = im * traceless_antihermitian(C23)
        C24 = plaquette(U, 2, 4, site)
        F[2, 4, site] = im * traceless_antihermitian(C24)
        C34 = plaquette(U, 3, 4, site)
        F[3, 4, site] = im * traceless_antihermitian(C34)
    end

    return nothing
end

function fieldstrength_eachsite!(
    ::Clover, F::Tensorfield{CPU,T}, U::Gaugefield{CPU,T}
) where {T}
    @assert dims(F) == dims(U)
    fac = Complex{T}(im / 4)

    @batch for site in eachindex(U)
        C12 = clover_square(U, 1, 2, site, 1)
        F[1, 2, site] = fac * traceless_antihermitian(C12)
        C13 = clover_square(U, 1, 3, site, 1)
        F[1, 3, site] = fac * traceless_antihermitian(C13)
        C14 = clover_square(U, 1, 4, site, 1)
        F[1, 4, site] = fac * traceless_antihermitian(C14)
        C23 = clover_square(U, 2, 3, site, 1)
        F[2, 3, site] = fac * traceless_antihermitian(C23)
        C24 = clover_square(U, 2, 4, site, 1)
        F[2, 4, site] = fac * traceless_antihermitian(C24)
        C34 = clover_square(U, 3, 4, site, 1)
        F[3, 4, site] = fac * traceless_antihermitian(C34)
    end

    return nothing
end

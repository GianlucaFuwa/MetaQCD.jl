abstract type AbstractFieldstrength end

struct Plaquette <: AbstractFieldstrength end
struct Clover <: AbstractFieldstrength end
struct Improved <: AbstractFieldstrength end

struct Tensorfield{D,T,A<:AbstractArray{SU{3,9,T}, 6}} <: Abstractfield{D,T,A}
    U::A
	NX::Int64
	NY::Int64
	NZ::Int64
	NT::Int64
	NV::Int64
	NC::Int64
end

function Tensorfield(NX, NY, NZ, NT; backend=CPU(), T=Val(Float64))
	D = if backend isa CPU
		CPUD
	elseif backend isa GPU
		GPUD
	else
		throw(AssertionError("Only CPU, CUDABackend or ROCBackend supported!"))
	end

	Tu = _unwrap_val(T)
	U = KA.zeros(backend, SMatrix{3,3,Complex{Tu},9}, 4, 4, NX, NY, NZ, NT)
	NV = NX * NY * NZ * NT
	NC = 3
	return Tensorfield{D,Tu,typeof(U)}(U, NX, NY, NZ, NT, NV, NC)
end

function Tensorfield(u::Abstractfield{D,T,A}) where {D,T,A}
    backend = get_backend(u)
    return Tensorfield(u.NX, u.NY, u.NZ, u.NT; backend=backend, T=Val(T))
end

@inline function Base.setindex!(f::T, v, μ, ν, x, y, z, t) where {T<:Tensorfield}
	f.U[μ,ν,x,y,z,t] = v
	return nothing
end

@inline function Base.setindex!(f::T, v, μ, ν, site::SiteCoords) where {T<:Tensorfield}
	f.U[μ,ν,site] = v
	return nothing
end

Base.getindex(f::T, μ, ν, x, y, z, t) where {T<:Abstractfield} = f.U[μ,ν,x,y,z,t]
Base.getindex(f::T, μ, ν, site::SiteCoords) where {T<:Abstractfield} = f.U[μ,ν,site]

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

function fieldstrength_eachsite!(::Plaquette, F::Tensorfield{CPUD}, U::Gaugefield{CPUD})
    @assert size(F) == size(U)

    @threads for site in eachindex(U)
        C12 = plaquette(U, 1, 2, site)
        F[1,2,site] = im * traceless_antihermitian(C12)
        C13 = plaquette(U, 1, 3, site)
        F[1,3,site] = im * traceless_antihermitian(C13)
        C14 = plaquette(U, 1, 4, site)
        F[1,4,site] = im * traceless_antihermitian(C14)
        C23 = plaquette(U, 2, 3, site)
        F[2,3,site] = im * traceless_antihermitian(C23)
        C24 = plaquette(U, 2, 4, site)
        F[2,4,site] = im * traceless_antihermitian(C24)
        C34 = plaquette(U, 3, 4, site)
        F[3,4,site] = im * traceless_antihermitian(C34)
    end

    return nothing
end

function fieldstrength_eachsite!(::Clover, F::Tensorfield{CPUD}, U::Gaugefield{CPUD})
    @assert size(F) == size(U)

    @threads for site in eachindex(U)
        C12 = clover_square(U, 1, 2, site, 1)
        F[1,2,site] = im/4 * traceless_antihermitian(C12)
        C13 = clover_square(U, 1, 3, site, 1)
        F[1,3,site] = im/4 * traceless_antihermitian(C13)
        C14 = clover_square(U, 1, 4, site, 1)
        F[1,4,site] = im/4 * traceless_antihermitian(C14)
        C23 = clover_square(U, 2, 3, site, 1)
        F[2,3,site] = im/4 * traceless_antihermitian(C23)
        C24 = clover_square(U, 2, 4, site, 1)
        F[2,4,site] = im/4 * traceless_antihermitian(C24)
        C34 = clover_square(U, 3, 4, site, 1)
        F[3,4,site] = im/4 * traceless_antihermitian(C34)
    end

    return nothing
end

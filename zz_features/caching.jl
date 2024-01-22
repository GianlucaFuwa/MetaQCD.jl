using BenchmarkTools
using Cthulhu
using CPUSummary
using Static
using StaticArrays
using MetaQCD
using MetaQCD.Utils
using MetaQCD.Gaugefields: AbstractGaugeAction
using OffsetArrays
using TiledIteration

function stencil_radius(::Type{GA}) where {GA}
    radius = GA==WilsonGaugeAction ? 1 : 2
    return radius
end

struct StaticGaugefield{Dims<:NTuple{4,Integer},GA<:AbstractGaugeAction}
	# Vector of 4D-Arrays performs better than 5D-Array for some reason
	U::Vector{OffsetArray{SMatrix{3, 3, ComplexF64, 9}, 4}}
	dims::Dims
	NV::Int64
	NC::Int64

    gaction::Type{GA}
	β::Float64
	Sg::Base.RefValue{Float64}
	CV::Base.RefValue{Float64}

	function StaticGaugefield(dims::NTuple{4,Integer}, β; GA=WilsonGaugeAction)
        rad = stencil_radius(GA)
		U = Vector{Array{SMatrix{3, 3, ComplexF64, 9}, 4}}(undef, 4)
        Udims = ntuple(i -> (1-rad):(dims[i]+rad), Val(4))

		for μ in static(1):static(4)
			U[μ] = Array{SMatrix{3, 3, ComplexF64, 9}, 4}(undef, dims)
			fill!(U[μ], zero3)
		end

		NV = prod(dims)

		Sg = Base.RefValue{Float64}(0.0)
		CV = Base.RefValue{Float64}(0.0)
		return new{typeof(dims),GA}(U, dims, NV, 3, GA, β, Sg, CV)
	end
end

function Base.show(io::IO, u::T) where {T<:StaticGaugefield}
    dims = size(u)
    print(io, "$(dims[1])x$(dims[2])x$(dims[3])x$(dims[4]) Gaugefield with $(u.gaction) and β = $(u.β)")
	return nothing
end

gauge_action(U::StaticGaugefield) = U.gaction
Base.similar(U::StaticGaugefield) =
    StaticGaugefield(gaugefield_dims(U), U.β; GA=gauge_action(U))
Base.size(U::StaticGaugefield) = U.dims
function Base.size(U::StaticGaugefield, ::Val{N}) where {N}
    1<=N<=4 || return 0
    return U.dims[N]
end

U = StaticGaugefield((12, 12, 12, 12), 6.0)
Ug = Gaugefield(12, 12, 12, 12, 6.0)
@generated function Base.size(U::Gaugefield, ::Val{N}) where {N}
    N==1 && return :(U.NX)
    N==2 && return :(U.NZ)
    N==3 && return :(U.NY)
    N==4 && return :(U.NT)
end

function task_local_memory(sc)::Vector{UInt8}
    (get!(task_local_storage(), _type_sym(sc)) do
        UInt8[]
    end)::Vector{UInt8}
end

function tile_gaugefield(U::StaticGaugefield; tiles_per_core=2)
    # number of blocks = size of the gaugefield in bytes / size of L2-cache
    # gaugefield_size = 144*4*prod(size(U))
    num_cores = min(CPUSummary.num_cores(), size(U, Val(4)))
    # num_blocks = cld(gaugefield_size, cache_size(Val(2)))
    num_tiles = num_cores * tiles_per_core
    tile_axes = SplitAxes(axes(U), num_tiles)
end

function plaquette_sum(U::Gaugefield)
    p = zeros(Float64, 8Threads.nthreads())

    for site in CartesianIndices(size(U))
        for μ in 1:3
            for ν in μ+1:4
                p[8Threads.threadid()] += plaquette(U, μ, ν, site)
            end
        end
    end

    return p
end

@inline function plaquette(U, μ, ν, site)
    Nμ = size(U)[1+μ]
    Nν = size(U)[1+ν]
    siteμ⁺ = move(site, μ, 1, Nμ)
    siteν⁺ = move(site, ν, 1, Nν)
    return remultr(U[μ,site], U[ν,siteμ⁺], U[μ,siteν⁺], U[ν,site])
end

function padded_tilesize1(::Type{T}, kernelsize::Dims, ncache = 2) where T
    nd = max(1, sum(x->x>1, kernelsize))
    # don't be too minimalist on the cache-friendly dim (use at least 2 cachelines)
    dim1minlen = 2*64÷sizeof(T)
    psz = (max(kernelsize[1], dim1minlen), Base.tail(kernelsize)...)
    L = sizeof(T)*prod(psz)
    # try to stay in L1 cache, but in the end we want a reasonably
    # favorable work ratio. f is the constant of proportionality in
    #      s+kernelsize ∝ kernelsize
    f = max(floor(Int, (cache_size(Val(2))/(ncache*L))^(1/nd)), 2)
    return TiledIteration._padded_tilesize_scale(f, psz)
end

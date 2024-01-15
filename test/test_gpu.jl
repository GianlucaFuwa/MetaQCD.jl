using MetaQCD
using MetaQCD.Utils
using BenchmarkTools
using CUDA
using Random
using StaticArrays
using KernelAbstractions
using Polyester
using LinearAlgebra

# Random.seed!(1206)

# L = (12, 12, 12, 12)
# β = 6.0
# gaction = WilsonGaugeAction

# Ucpu1 = initial_gauges("cold", L..., β, type_of_gaction=gaction);
# Ucpu2 = Array{SMatrix{3, 3, ComplexF64, 9}, 5}(undef, 4, L...); fill!(Ucpu2, MetaQCD.eye3);
# Ugpu = CuArray{SMatrix{3, 3, ComplexF64, 9}, 5}(undef, 4, L...); fill!(Ugpu, MetaQCD.eye3);

@kernel function plaquette_kernel!(out, @Const(U))
	# Get block / workgroup size
    gs = @uniform @groupsize()
	numthreads = gs[1] * gs[2] * gs[3] * gs[4] * gs[5]
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	μ, ix, iy, iz, it = @index(Global, NTuple)
	site = SiteCoords(ix, iy, iz, it)
	# Create local block on shared memory including boundaries
	# subU = @localmem eltype(U) (gs[1]+2, gs[2]+2, gs[3]+2, gs[4]+2)


    # Create local cache for each thread in workgroup and perform operation
    cache = @localmem eltype(out) (numthreads,)
	ti = @index(Local, Linear)

	p = 0.0
	for ν in μ+1:4
		Nμ = size(U)[μ+1]
		Nν = size(U)[ν+1]
		siteμ⁺ = move(site, μ, 1, Nμ)
		siteν⁺ = move(site, ν, 1, Nν)
		plaq = cmatmul_oodd(U[μ,site], U[ν,siteμ⁺], U[μ,siteν⁺], U[ν,site])
		p += real(tr(plaq))
	end
	cache[ti] = p
    @synchronize

    # Reduce elements in shared memory using sequential addressing following
    # https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    @private s = div(numthreads, 2)
    while s > 0
        if ti < s
			cache[ti] += cache[ti + s]
        end

        @synchronize
        s = s >> 1
    end

	if ti == 1
		out[bi] = cache[1]
	end
end

function plaquette_sum(U)
    backend = get_backend(U)
	ndrange = (3, size(U)[2:end]...)
	workgroupsize = (3, 4, 4, 4, 4)
	numelements = prod(ndrange)
	numblocks = cld(numelements, prod(workgroupsize))

	out_type = U |> eltype |> eltype |> real

	out = zeros(out_type, numblocks) |> (backend isa CUDABackend ? CuArray : Array)
    kernel = plaquette_kernel!(backend, workgroupsize)
    kernel(out, U, ndrange=size(U))
	KernelAbstractions.synchronize(backend)
	return sum(out)
end

# ndrange = size(Ugpu)
# workgroupsize = (4, 4, 4, 4, 4)

# iterspace, d = KernelAbstractions.partition(plaquette_kernel!(CUDABackend(), workgroupsize), ndrange, nothing)
# KernelAbstractions.NDIteration.blocks(iterspace)
# KernelAbstractions.NDIteration.workitems(iterspace)

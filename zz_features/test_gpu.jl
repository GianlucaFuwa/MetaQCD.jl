using MetaQCD
using MetaQCD.Utils
using BenchmarkTools
using CUDA
using Random
using StaticArrays
using KernelAbstractions
using Polyester
using LinearAlgebra

Random.seed!(1206)

L = (12, 12, 12, 12)
β = 6.0
gaction = WilsonGaugeAction

# Ucpu1 = initial_gauges("cold", L..., β, type_of_gaction=gaction);
Ucpu = Array{SMatrix{3, 3, ComplexF64, 9}, 5}(undef, 4, L...); fill!(Ucpu, eye3);
Ugpu = CuArray{SMatrix{3, 3, ComplexF32, 9}, 5}(undef, 4, L...); fill!(Ugpu, ComplexF16.(eye3));

@kernel function plaquette_kernel!(out, @Const(U))
	# Get block / workgroup size
    gs = @uniform @groupsize()
	# workgroup index, that we use to pass the reduced value to global "out"
	bi = @index(Group, Linear)
	site = @index(Global, Cartesian)

    # Create local cache for each thread in workgroup and perform operation
    cache = @localmem eltype(out) (prod(gs),)
	ti = @index(Local, Linear)

	p = 0.0
	KernelAbstractions.Extras.LoopInfo.@unroll for μ in 1:3
		for ν in μ+1:4
			Nμ = size(U)[μ+1]
			Nν = size(U)[ν+1]
			siteμ⁺ = move(site, μ, 1, Nμ)
			siteν⁺ = move(site, ν, 1, Nν)
			plaq = cmatmul_oodd(U[μ,site], U[ν,siteμ⁺], U[μ,siteν⁺], U[ν,site])
			p += real(tr(plaq))
		end
	end
	cache[ti] = p
    @synchronize

    # Reduce elements in shared memory using sequential addressing following
    # https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    # @private s = div(length(cache), 2)
    # while s > 0
    #     if ti < s
	# 		cache[ti] += cache[ti + s]
    #     end

    #     @synchronize
    #     s = s >> 1
    # end

	if ti == 1
		out[bi] = cache[1]
	end
end

@kernel function kern()
	site = @index(Global, Cartesian)
	@print "site = $site\n"
end

function plaquette_sum(U)
    backend = get_backend(U)
	ndrange = size(U)[2:end]
	workgroupsize = (4, 4, 4, 4)
	numelements = prod(ndrange)
	numblocks = cld(numelements, prod(workgroupsize))

	out_type = U |> eltype |> eltype |> real

	out = KernelAbstractions.zeros(backend, out_type, numblocks)
    kernel = plaquette_kernel!(backend, workgroupsize)
    kernel(out, U, ndrange=ndrange)
	KernelAbstractions.synchronize(backend)
	return Float64(reduce(+, out))
end

ndrange = size(Ucpu)[2:end]
workgroupsize = (4, 4, 4, 4)

iterspace, d = KernelAbstractions.partition(plaquette_kernel!(CUDABackend(), workgroupsize), ndrange, nothing)
KernelAbstractions.NDIteration.blocks(iterspace)
KernelAbstractions.NDIteration.workitems(iterspace)

@btime plaquette_sum($Ucpu)
@btime plaquette_sum($Ugpu)
# @btime plaquette_sum($Ugpu)

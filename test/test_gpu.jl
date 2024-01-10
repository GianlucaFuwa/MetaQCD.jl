using MetaQCD
using MetaQCD.Utils
using BenchmarkTools
using CUDA
using Random
using StaticArrays
using KernelAbstractions
using Polyester

Random.seed!(1206)

L = (12, 12, 12, 12)
β = 6.0
gaction = WilsonGaugeAction

Ucpu1 = Gaugefield(L..., β, GA=gaction);
Ucpu2 = Array{SMatrix{3, 3, ComplexF64, 9}, 5}(undef, 4, L...); fill!(Ucpu2, MetaQCD.zero3);
Ugpu = CuArray{SMatrix{3, 3, ComplexF64, 9}, 5}(undef, 4, L...); fill!(Ugpu, MetaQCD.zero3);

@kernel function plaquette_kernel!(out, @Const(U))
	# TODO get index of group gi = @index(Group, Linear)
    μ, ix, iy, iz, it = @index(Global, NTuple)
	site = SiteCoords(ix, iy, iz, it)

	p = zero(eltype(out))
	@inbounds for μ in 1:4
		for ν in μ+1:4
			Nμ = size(U)[μ+1]
    		Nν = size(U)[ν+1]
			siteμ⁺ = move(site, μ, 1, Nμ)
    		siteν⁺ = move(site, ν, 1, Nν)
    		plaq = cmatmul_oodd(U[μ,site], U[ν,siteμ⁺], U[μ,siteν⁺], U[ν,site])
			p += real(tr(plaq))
		end
	end

	out[gi] += p
end

function plaquette_sum(U, ϵ)
    backend = get_backend(U)

    kernel = random_gaugess_kernel!(backend)
    kernel(U, ϵ, ndrange=size(U))
end

function random_gauges!(u::T, ϵ) where {T<:Gaugefield}
	@batch for site = eachindex(u)
		for μ in 1:4
			u[μ][site] = gen_SU3_matrix(ϵ)
		end
	end

	return nothing
end

struct SU{N} end
Base.in(g, ::Type{SU{3}}) = is_special_unitary(g)
@show is_special_unitary(Ucpu[1])
# A = KernelAbstractions.zeros(backend, Float64, 4, 12, 12, 12, 12);

@benchmark random_gauges!($U, 0.1)
@benchmark random_gaugess!($Ucpu, 0.1)
@benchmark CUDA.@sync random_gaugess!($Ugpu, 0.1)

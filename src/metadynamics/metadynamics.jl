module Metadynamics
	using ..SystemParameters: Params
	
	import ..AbstractMeasurementModule: top_charge
	import ..AbstractSmearingModule: StoutSmearing
	import ..Gaugefields: AbstractGaugeAction, Gaugefield

	struct MetaEnabled end
	struct MetaDisabled end
	
	struct BiasPotential
		kind_of_cv::String
		smearing::StoutSmearing{<:AbstractGaugeAction}
		symmetric::Bool

		CVlims::NTuple{2, Float64}
		bin_width::Float64
		weight::Float64
		penalty_weight::Float64
		is_static::Bool
		
		values::Vector{Float64}
		bin_vals::Vector{Float64}
		# exceeded_count::Int64
		fp::Union{Nothing, IOStream}
		
		function BiasPotential(p::Params, instance = 1)
			NX, NY, NZ, NT = p.L
			_temp_U = Gaugefield(NX, NY, NZ, NT, p.β, kind_of_gaction = p.kind_of_gaction)
			smearing = StoutSmearing(_temp_U, p.smearing_for_cv, p.ρstout_for_cv)

			if instance == 0
				values = potential_from_file(p, nothing) 
			else
				values = potential_from_file(p, p.usebias[instance])
			end

			bin_vals = range(p.CVlims[1], p.CVlims[2], step = p.bin_width)
			# exceeded_count = 0
			fp = instance == 0 ? nothing : open(p.biasfile[instance], "w")

			return new(
				p.kind_of_cv, smearing, p.symmetric,
				p.CVlims, p.bin_width, p.bias_weight, p.penalty_weight, p.is_static,
				values, bin_vals, fp,
			)
		end

		function BiasPotential(
			U,
			kind_of_cv,
			numsmear,
			ρstout,
			symmetric,
			CVlims,
			bin_width,
			weight,
			penalty_weight,			
		)	
			smearing = StoutSmearing(U, numsmear, ρstout)
			is_static = false
			values = zeros(round(Int64, (CVlims[2]-CVlims[1]) / bin_width, RoundNearestTiesAway) + 1)
			bin_vals = range(CVlims[1], CVlims[2], step = bin_width)
			# exceeded_count = 0
			fp = nothing

			return new(
				kind_of_cv, smearing, symmetric,
				CVlims, bin_width, weight, penalty_weight, is_static,
				values, bin_vals, fp,
			)
		end

	end
	
	function potential_from_file(p::Params, usebias)
		if usebias === nothing
			return zeros(round(Int64, (p.CVlims[2]-p.CVlims[1]) / p.bin_width, RoundNearestTiesAway) + 1)
		else
			values = readdlm(usebias, Float64)
			@assert length(values[:,2]) == round(
				Int64,
				(p.CVlims[2]-p.CVlims[1]) / p.bin_width,
				RoundNearestTiesAway,
			) "Length of passed Metapotential doesn't match parameters"
			return values[:,2]
		end
	end

	function Base.length(b::BiasPotential)
		return length(b.values)
	end

	function Base.flush(b::BiasPotential)
		if b.fp !== nothing
			flush(b.fp)
		end
	end

	function Base.seekstart(b::BiasPotential)
		if b.fp !== nothing
			seekstart(b.fp)
		end
	end

	function Base.setindex!(b::BiasPotential, v, i)
		b.values[i] = v
	end

	@inline function Base.getindex(b::BiasPotential, i)
		return b.values[i]
	end

	@inline function index(b::BiasPotential, cv)
		grid_index = (cv - b.CVlims[1]) / b.bin_width + 0.5
		return round(Int64, grid_index, RoundNearestTiesAway)
	end

	function update_bias!(b::BiasPotential, cv)
		grid_index = index(b, cv)

		if 1 <= grid_index <= length(b.values)
			for (idx, current_bin) in enumerate(b.bin_vals)
				b[idx] += b.weight * exp(-0.5(cv - current_bin)^2 / b.bin_width^2)
			end	
		else
			# b.exceeded_count += 1
		end

		return nothing
	end
	
	function (b::BiasPotential)(cv)
		return return_potential(b, cv)
	end

	function return_potential(b::BiasPotential, cv)
		if b.CVlims[1] <= cv < b.CVlims[2]
			grid_index = index(b, cv)
			return b[grid_index]
		else
			penalty = b.k * (0.1 + min((cv - b.CVlims[1])^2, (cv - b.CVlims[2])^2))
			return penalty
		end
	end

	"""
	Approximate ∂V/∂Q by use of the five-point stencil
	""" 
	function ∂V∂Q(b::BiasPotential, cv)
		bin_width = b.bin_width
		num = -b(cv + 2 * bin_width) + 
			8 * b(cv + bin_width) - 
			8 * b(cv - bin_width) + 
			b(cv - 2 * bin_width)
		denom = 12 * bin_width
		return num / denom
	end

end
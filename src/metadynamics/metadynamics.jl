module Metadynamics
	using ..SystemParameters: Params
	
	import ..AbstractMeasurementModule: top_charge
	import ..Gaugefields: Gaugefield
	
	mutable struct BiasPotential
		kind_of_cv::String
		smears_for_cv::Int64
		ρstout_for_cv::Float64
		_Usmeared::Gaugefield
		symmetric::Bool
		CVlims::NTuple{2, Float64}
		bin_width::Float64
		bias_weight::Float64
		penalty_weight::Float64
		is_static::Bool
		
		values::Vector{Float64}
		bin_vals::Vector{Float64}
		exceeded_count::Int64
		fp::Union{Nothing, IOStream}
		
		function BiasPotential(p::Params, instance = 1)
			kind_of_cv = p.kind_of_cv
			smears_for_cv = p.smears_for_cv
			ρstout_for_cv = p.ρstout_for_cv
			symmetric = p.symmeteric
			CVlims = p.CVlims
			bin_width = p.bin_width
			bias_weight = p.bias_weight
			penalty_weight = p.penalty_weight
			is_static = instance == 0 ? true : p.is_static[instance]

			if instance == 0
				values = potential_from_file(p, nothing) 
			else
				values = potential_from_file(p, p.usebias[instance])
			end

			bin_vals = range(CVlims[1], CVlims[2], step = bin_width)
			exceeded_count = 0
			fp = instance == 0 ? nothing : open(p.biasfile[instance], "w")

			return new(kind_of_cv, smears_for_cv, ρstout_for_cv, symmetric, CVlims,
			bin_width, bias_weight, penalty_weight, is_static, values, bin_vals,
			exceeded_count, fp)
		end	
	end
	
	function potential_from_file(p::Params, usebias)
		if usebias === nothing
			return zeros(round(Int64, (p.CVlims[2]-p.CVlims[1]) / p.bin_width, RoundNearestTiesAway))
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
		grid_index = index(b,cv)

		if 1 <= grid_index <= length(b.values)
			for (idx,current_bin) in enumerate(b.cv_vals)
				b[idx] += b.w * exp(-0.5(cv-current_bin)^2 / b.bin_width^2)
			end	
		else
			b.exceeded_count += 1
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
	Approximate dV/dQ by use of the five-point stencil
	""" 
	function return_derivative(b::BiasPotential, cv)
		bin_width = b.bin_width
		num = -return_potential(b, cv + 2 * bin_width) + 
			8 * return_potential(b, cv + bin_width) - 
			8 * return_potential(b, cv - bin_width) + 
			return_potential(b, cv - 2 * bin_width)
		denom = 12 * bin_width
		return num / denom
	end
	
	function (bias::BiasPotential)(U::Gaugefield)
		smearing = StoutSmearing(bias.smears_for_cv, bias.ρstout_for_cv)
		Usmeared = deepcopy(U)
		for i in 1:bias.smears_for_cv
			
		calc_gauge_force!(force_toplayer, staples, Utmp)
		force_bottomlayer = stout_recursion(
			force_toplayer,
			Uout_multi,
			staples_multi,
			Qs_multi,
			smearing,
		)
		topcharge = top_charge(Utmp, Bias.kind_of_cv)
		end
	end

end



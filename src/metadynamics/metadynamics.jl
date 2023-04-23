module Metadynamics
	using Printf
	using ..System_parameters:Params
	
	mutable struct Bias_potential
		kind_of_CV::String
		smears_for_CV::Int64
		ρstout_for_CV::Float64
		symmetric::Bool
		CVlims::NTuple{2,Float64}
		bin_width::Float64
		bias_weight::Float64
		penalty_weight::Float64
		is_static::Bool
		sq_param::Float64
		
		values::Array{Float64,1}
		bin_vals::Array{Float64,1}
		exceeded_count::Int64
		fp::Union{Nothing,IOStream}
		
		function Bias_potential(p::Params, instance=1)
			kind_of_CV = p.kind_of_CV
			smears_for_CV = p.smears_for_CV
			ρstout_for_CV = p.ρstout_for_CV
			symmetric = p.symmeteric
			CVlims = p.CVlims
			bin_width = p.bin_width
			bias_weight = p.bias_weight
			penalty_weight = p.penalty_weight
			is_static = instance==0 ? true : p.is_static[instance]
			sq_param = instance==0 ? 0.0 : p.sq_param[instance]

			values = instance==0 ? potential_from_file(p,nothing) : potential_from_file(p,p.usebias[instance])
			bin_vals = range(CVlims[1],CVlims[2],step=bin_width)
			exceeded_count = 0
			fp = instance==0 ? nothing : open(p.biasfile[instance],"w")

			return new(kind_of_CV,smears_for_CV,ρstout_for_CV,symmetric,CVlims,bin_width,bias_weight,penalty_weight,is_static,sq_param,
			values,bin_vals,exceeded_count,fp)
		end	
	end
	
	function potential_from_file(p::Params,usebias::Union{Nothing,String})
		if usebias === nothing
			return zeros(round(Int,(p.CVlims[2]-p.CVlims[1])/p.bin_width,RoundNearestTiesAway))
		else
			values = readdlm(usebias,Float64)
			@assert length(values[:,2]) == round(Int,(p.CVlims[2]-p.CVlims[1])/p.bin_width,RoundNearestTiesAway)
			return values[:,2]
		end
	end

	function get_kind_of_CV(b::Bias_potential)
		return b.kind_of_CV
	end

	function get_smearparams_for_CV(b::Bias_potential)
		return b.smears_for_CV, b.ρstout_for_CV
	end

	function is_static(b::Bias_potential)
		return b.is_static
	end

	function Base.length(b::Bias_potential)
		return length(b.values)
	end

	function get_binwidth(b::Bias_potential)
		return b.bin_width
	end

	function Base.flush(b::Bias_potential)
		if b.fp !== nothing
			flush(b.fp)
		end
	end

	function Base.seekstart(b::Bias_potential)
		if b.fp !== nothing
			seekstart(b.fp)
		end
	end

	function Base.setindex!(b::Bias_potential,v,i)
		b.values[i] = v
	end

	@inline function Base.getindex(b::Bias_potential,i)
		return b.values[i]
	end

	@inline function index(b::Bias_potential,cv::Float64)
		grid_index = (cv-b.CVlims[1])/b.bin_width + 0.5
		return round(Int,grid_index,RoundNearestTiesAway)
	end

	function update_bias!(b::Bias_potential,cv::Float64)
		grid_index = index(b,cv)
		if 1 <= grid_index <= length(b.values)
			for (idx,current_bin) in enumerate(b.cv_vals)
				b[idx] += b.w*exp(-0.5(cv-current_bin)^2/b.bin_width^2)
			end	
		else
			b.exceeded_count += 1
		end
		return nothing
	end

	function ReturnPotential(b::Bias_potential, cv::Float64)
		if b.CVlims[1] <= cv < b.CVlims[2]
			grid_index = index(b,cv)
			return b[grid_index]
		else
			penalty = b.k*(0.1+min((cv-b.CVlims[1])^2,(cv-b.CVlims[2])^2))
			return penalty
		end
	end

	"""
	Approximate dV/dQ by use of the five-point stencil
	""" 
	function ReturnDerivative(b::Bias_potential, cv::Float64)
		bin_width = get_binwidth(b)

		num = -ReturnPotential(b, cv + 2*bin_width) + 
			 8*ReturnPotential(b, cv + bin_width)   - 
			 8*ReturnPotential(b, cv - bin_width)   + 
			   ReturnPotential(b, cv - 2*bin_width)
		denom = 12 * bin_width
		return num / denom
	end

	function DeltaV(b::Bias_potential ,cvold::Float64, cvnew::Float64)
		dV = ReturnPotential(b,cvnew) - ReturnPotential(b,cvold)
		dV_sq = b.sq_param * (cvnew^2-cvold^2)
		return dV + dV_sq
	end

end



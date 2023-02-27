module Metadynamics
	using Printf
	using ..System_parameters:Params
	
	mutable struct Bias_potential
		symmetric::Bool
		CVlims::NTuple{2,Float64}
		CVthr::NTuple{2,Float64}
		bin_width::Float64
		bias_weight::Float64
		penalty_weight::Float64
		is_static::Bool
		sq_param::Float64
		
		cv_storage::Array{Float64,1}
		values::Array{Float64,1}
		bin_vals::Array{Float64,1}
		exceeded_count::Int64
		fp::IOStream
		
		function Bias_potential(p::Params)
			symmetric = p.symmeteric
			CVlims = p.CVlims
			CVthr = p.CVthr
			bin_width = p.bin_width
			bias_weight = p.bias_weight
			penalty_weight = p.penalty_weight
			is_static = p.is_static[instance]
			sq_param = p.sq_param[instance]

			cv_storage = zeros(Float64,p.Nsweeps)
			values = potential_from_file(p,p.usebias[instance])
			bin_vals = range(Qlims[1],Qlims[2]-bin_width,step=bin_width)
			exceeded_count = 0
			fp = open(p.biasfile[instance],"w")

			return new(symmetric,CVlims,CVthr,bin_width,bias_weight,penalty_weight,is_static,sq_param,
			cv_storage,values,bin_vals,exceeded_count,fp)
		end	
	end
	
	function potential_from_file(p::Params,usebias::Union{Nothing,String})
		if usebias === nothing
			return zeros(round(Int,(p.Qlims[2]-p.Qlims[1])/p.bin_width,RoundNearestTiesAway))
		else
			values = readdlm(usebias,Float64)
			@assert length(values[:,2]) == round(Int,(p.Qlims[2]-p.Qlims[1])/p.bin_width,RoundNearestTiesAway)
			return values[:,2]
		end
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
		grid_index = (cv-b.Qlims[1])/b.bin_width + 0.5
		return round(Int,grid_index,RoundNearestTiesAway)
	end

	function update_bias!(b::Bias_potential,cv::Float64)
		cv = b.symmetric ? abs(cv) : cv
		grid_index = index(b,cv)
		if 1 < grid_index < length(b.values)
			grid_cv = b.bin_vals[grid_index]
			b[grid_index]   += b.bias_weight*(1-(cv-grid_cv)/b.bin_width)
			b[grid_index+1] += b.bias_weight*(cv-grid_cv)/b.bin_width
		elseif grid_index == 1
			grid_cv = b.Qlims[1]
			b[grid_index] +=  b.bias_weight*(1-(cv-grid_cv)/b.bin_width)
		elseif grid_index == length(b.values)
			grid_cv = b.Qlims[2]-b.bin_width
			b[grid_index] +=  b.bias_weight*(1-(cv-grid_cv)/b.bin_width)
		else
			b.exceeded_count += 1
		end
		return nothing
	end

	function ReturnPotential(b::Bias_potential,cv::Float64)
		cv = b.symmetric ? abs(cv) : cv
		grid_index = index(b,cv)
		if b.Qthr[1] ≤ cv ≤ b.Qthr[2]
			return b[grid_index]
		else 
			penalty = b.penalty_weight*( 0.1 + min( (cv-b.Qthr[1])^2, (cv-b.Qthr[2])^2 ) )
			return penalty
		end
	end

	function DeltaV(b::Bias_potential,cvold::Float64,cvnew::Float64)
		dV = ReturnPotential(b,cvnew) - ReturnPotential(b,cvold)
		dV_sq = b.sq_param*(cvnew^2-cvold^2)
		return dV + dV_sq
	end

end



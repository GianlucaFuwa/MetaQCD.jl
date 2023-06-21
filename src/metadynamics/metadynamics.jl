module Metadynamics
    using Polyester
	using ..SystemParameters: Params

	import ..AbstractMeasurementModule: top_charge
	import ..AbstractSmearingModule: StoutSmearing
	import ..Gaugefields: AbstractGaugeAction, Gaugefield

	struct MetaEnabled end
	struct MetaDisabled end

    abstract type AbstractBiasPotential end

    include("biaspotential.jl")

	function potential_from_file(p::Params, usebias)
		if usebias === nothing
			return zero(range(p.CVlims[1], p.CVlims[2], step = p.bin_width))
		else
			values = readdlm(usebias, Float64)
			@assert length(values[:, 2]) == round(
				Int64,
				(p.CVlims[2] - p.CVlims[1]) / p.bin_width,
				RoundNearestTiesAway,
			) "Length of passed Metapotential doesn't match parameters"
			return values[:,2]
		end
	end

end

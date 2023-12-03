module Universe

using Dates
using Unicode
using ..Utils
using ..Output

import ..Gaugefields: Gaugefield, Plaquette, Clover
import ..Gaugefields: AbstractGaugeAction, DBW2GaugeAction, IwasakiGaugeAction,
    SymanzikTadGaugeAction, SymanzikTreeGaugeAction, WilsonGaugeAction
import ..Gaugefields: initial_gauges
import ..BiasModule: Bias, get_cvtype_from_parameters, write_to_file
import ..Parameters: ParameterSet
import ..Smearing: NoSmearing, StoutSmearing

struct Univ{TG,TB}
    U::TG
    bias::TB
    numinstances::Int64
end

function Univ(p::ParameterSet; use_mpi=false, fp=true)
    @level1("┌ Setting Universe...")
    NX, NY, NZ, NT = p.L
    β = p.beta
    GA = get_gaugeaction_from_parameters(p)

    @level1("|  L: $(NX)x$(NY)x$(NZ)x$(NT)")
    @level1("|  GaugeAction: $(GA)")
    @level1("|  beta: $β")
    @level1("└\n")

    if p.kind_of_bias != "none"
        if p.tempering_enabled && use_mpi==false
            numinstances = p.numinstances
            @level1("|  Using 1 + $(numinstances-1) instances\n")
            U₁ = initial_gauges(p.initial, NX, NY, NZ, NT, β, type_of_gaction=GA)
            bias₁ = Bias(p, U₁; instance=0)

            U = Vector{typeof(U₁)}(undef, numinstances)
            bias = Vector{typeof(bias₁)}(undef, numinstances)

            bias[1] = bias₁
            U[1] = U₁
            for i in 2:numinstances
                U[i] = initial_gauges(p.initial, NX, NY, NZ, NT, β, type_of_gaction=GA)
                bias[i] = Bias(p, U[i]; instance=i-1)
            end
        else
            numinstances = 1
            U = initial_gauges(p.initial, NX, NY, NZ, NT, β, type_of_gaction=GA)
            bias = Bias(p, U)
        end
    else
        @assert p.tempering_enabled == false "tempering can only be enabled with bias"
        numinstances = 1
        U = initial_gauges(p.initial, NX, NY, NZ, NT, β, type_of_gaction=GA)
        bias = nothing
    end

    return Univ{typeof(U), typeof(bias)}(U, bias, numinstances)
end

function get_gaugeaction_from_parameters(p::ParameterSet)
    kind_of_gaction = Unicode.normalize(p.kind_of_gaction, casefold=true)

    if kind_of_gaction == "wilson"
        GA = WilsonGaugeAction
    elseif kind_of_gaction == "symanzik_tree"
        GA = SymanzikTreeGaugeAction
    elseif kind_of_gaction == "symanzik_tadpole"
        GA = SymanzikTadGaugeAction
    elseif kind_of_gaction == "iwasaki"
        GA = IwasakiGaugeAction
    elseif kind_of_gaction == "dbw2"
        GA = DBW2GaugeAction
    else
        error("Gauge action '$(p.kind_of_gaction)' not supported")
    end

    return GA
end

end

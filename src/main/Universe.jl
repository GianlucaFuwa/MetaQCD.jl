module Universe

using Dates
using InteractiveUtils
using ..Utils
using ..Output

import ..Gaugefields: Gaugefield, Plaquette, Clover
import ..Gaugefields: AbstractGaugeAction, DBW2GaugeAction, IwasakiGaugeAction,
    SymanzikTadGaugeAction, SymanzikTreeGaugeAction, WilsonGaugeAction
import ..Gaugefields: initial_gauges
import ..BiasModule: Bias, get_cvtype_from_parameters, write_to_file
import ..Parameters: ParameterSet
import ..Smearing: NoSmearing, StoutSmearing

struct Univ{TG,TB,TV<:VerboseLevel}
    U::TG
    bias::TB
    numinstances::Int64
    verbose_print::TV
end

function Univ(p::ParameterSet; use_mpi=false, fp=true)
    NX, NY, NZ, NT = p.L
    β = p.beta
    verbose_print = get_verboseprint_from_parameters(p, fp)
    fp && println_verbose1(verbose_print, "# ", pwd())
    fp && println_verbose1(verbose_print, "# ", Dates.now())
    io = IOBuffer()

    InteractiveUtils.versioninfo(io)
    versioninfo = String(take!(io))
    fp && println_verbose1(verbose_print, versioninfo)

    fp && println_verbose1(verbose_print, ">> PHYSICAL PARAMS: L = $NX×$NX×$NX×$NX, β = $β\n")

    GA = get_gaugeaction_from_parameters(p)

    if p.kind_of_bias != "none"
        if p.tempering_enabled && use_mpi == false
            numinstances = p.numinstances
            println_verbose1(verbose_print, ">> Using 1 + $(numinstances-1) instances\n")
            U₁ = initial_gauges(p.initial, NX, NY, NZ, NT, β, type_of_gaction=GA)
            bias₁ = Bias(p, U₁; instance=0, verbose=verbose_print, has_fp=fp)

            U = Vector{typeof(U₁)}(undef, numinstances)
            bias = Vector{typeof(bias₁)}(undef, numinstances)

            bias[1] = bias₁
            U[1] = U₁
            for i in 2:numinstances
                U[i] = initial_gauges(p.initial, NX, NY, NZ, NT, β, type_of_gaction=GA)
                bias[i] = Bias(p, U[i]; verbose=verbose_print, instance=i-1, has_fp=fp)
            end
        else
            vb = fp ? verbose_print : nothing
            numinstances = 1
            U = initial_gauges(p.initial, NX, NY, NZ, NT, β, type_of_gaction=GA)
            bias = Bias(p, U; verbose=vb)
        end
    else
        @assert p.tempering_enabled == false "tempering can only be enabled with bias"
        numinstances = 1
        U = initial_gauges(p.initial, NX, NY, NZ, NT, β, type_of_gaction=GA)
        bias = nothing
    end

    return Univ{typeof(U), typeof(bias), typeof(verbose_print)}(
        U,
        bias,
        numinstances,
        verbose_print,
    )
end

function get_gaugeaction_from_parameters(p::ParameterSet)
    if p.kind_of_gaction=="wilson"
        GA = WilsonGaugeAction
    elseif p.kind_of_gaction=="symanzik_tree"
        GA = SymanzikTreeGaugeAction
    elseif p.kind_of_gaction=="symanzik_tadpole"
        GA = SymanzikTadGaugeAction
    elseif p.kind_of_gaction=="iwasaki"
        GA = IwasakiGaugeAction
    elseif p.kind_of_gaction=="dbw2"
        GA = DBW2GaugeAction
    else
        error("Gauge action '$(p.kind_of_gaction)' not supported")
    end

    return GA
end

function get_verboseprint_from_parameters(p::ParameterSet, fp)
    if p.verboselevel == 1
        verbose_print = fp ? Verbose1(p.load_fp) : Verbose1()
    elseif p.verboselevel == 2
        verbose_print = fp ? Verbose2(p.load_fp) : Verbose2()
    elseif p.verboselevel == 3
        verbose_print = fp ? Verbose3(p.load_fp) : Verbose3()
    else
        error("Verbose level can only be 1, 2 or 3. Now it's $(p.verboselevel)")
    end

    return verbose_print
end

end

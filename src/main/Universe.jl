module Universe
    using ..Utils
    using ..Output

    import ..Gaugefields: Gaugefield, Plaquette, Clover
    import ..Gaugefields: AbstractGaugeAction, DBW2GaugeAction, IwasakiGaugeAction,
        SymanzikTadGaugeAction, SymanzikTreeGaugeAction, WilsonGaugeAction
    import ..Gaugefields: initial_gauges
    import ..BiasModule: Bias, get_cvtype_from_parameters, write_to_file
    import ..Parameters: ParameterSet
    import ..Smearing: NoSmearing, StoutSmearing

    struct Univ{TG,TB,TV}
        U::TG
        bias::TB
        numinstances::Int64
        verbose_print::TV
    end

    function Univ(p::ParameterSet; use_mpi=false, fp=true)
        NX, NY, NZ, NT = p.L

        if p.verboselevel == 1
            verbose_print = fp==true ? Verbose1(p.load_fp) : Verbose1()
        elseif p.verboselevel == 2
            verbose_print = fp==true ? Verbose2(p.load_fp) : Verbose2()
        elseif p.verboselevel == 3
            verbose_print = fp==true ? Verbose3(p.load_fp) : Verbose3()
        end

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
            error("Gauge action '$(kind_of_gaction)' not supported")
        end

        U = initial_gauges(p.initial, NX, NY, NZ, NT, p.beta, type_of_gaction=GA)

        if p.kind_of_bias != "none"
            if p.tempering_enabled && use_mpi == false
                numinstances = p.numinstances
                bias = Bias(p, U; instance=i-1, verbose=verbose_print, has_fp=fp)

                U = Vector{typeof(U)}(undef, numinstances)
                bias = Vector{typeof(bias)}(undef, numinstances)

                for i in 2:numinstances
                    U[i] = deepcopy(U₁)
                    vb = fp ? verbose_print : nothing
                    bias[i] = Bias(p, U[1]; verbose=vb, instance=i-1, has_fp=fp)
                end
            else
                numinstances = 1
                bias = Bias(p, U; verbose=verbose_print)
            end
        else
            @assert p.tempering_enabled == false "tempering can only be enabled with bias"
            numinstances = 1
            bias = nothing
        end

        return Univ{typeof(U), typeof(bias), typeof(verbose_print)}(
            U,
            bias,
            numinstances,
            verbose_print,
        )
    end

    function get_cvsmearing_from_parameters(p::ParameterSet)
        if p.numsmears_for_cv==0 || p.rhostout_for_cv==0
            return NoSmearing
        elseif p.numsmears_for_cv>0 && p.rhostout_for_cv>0
            return StoutSmearing
        else
            error("numsmears_for_cv and rhostout_for_cv must be >0")
        end
    end
end

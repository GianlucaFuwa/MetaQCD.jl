module Universe
    using ..Utils
    using ..Output

    import ..Gaugefields: Gaugefield, Plaquette, Clover
    import ..Gaugefields: AbstractGaugeAction, DBW2GaugeAction, IwasakiGaugeAction,
        SymanzikTadGaugeAction, SymanzikTreeGaugeAction, WilsonGaugeAction
    import ..Gaugefields: identity_gauges, random_gauges
    import ..Metadynamics: BiasPotential, get_cvtype_from_parameters, write_to_file
    import ..Parameters: ParameterSet
    import ..Smearing: NoSmearing, StoutSmearing

    struct Univ{TG,TB,TV}
        meta_enabled::Bool
        tempering_enabled::Bool
        U::TG
        Bias::TB
        numinstances::Int64
        verbose_print::TV
    end

    function Univ(p::ParameterSet; use_mpi=false, fp=true)
        NX, NY, NZ, NT = p.L

        if p.kind_of_gaction == "wilson"
            GA = WilsonGaugeAction
        elseif p.kind_of_gaction == "symanzik_tree"
            GA = SymanzikTreeGaugeAction
        elseif p.kind_of_gaction == "symanzik_tadpole"
            GA = SymanzikTadGaugeAction
        elseif p.kind_of_gaction == "iwasaki"
            GA = IwasakiGaugeAction
        elseif p.kind_of_gaction == "dbw2"
            GA = DBW2GaugeAction
        else
            error("Gauge action '$(kind_of_gaction)' not supported")
        end

        if p.meta_enabled
            tempering_enabled = p.tempering_enabled

            if tempering_enabled && use_mpi == false
                numinstances = p.numinstances
                TG = Gaugefield{GA}
                TCV = get_cvtype_from_parameters(p)
                cvsmearing = get_cvsmearing_from_parameters(p)
                TS = cvsmearing==NoSmearing ? cvsmearing : cvsmearing{TG}
                TB = BiasPotential{TCV,TG,TS}
                U = Vector{TG}(undef, p.numinstances)
                Bias = Vector{TB}(undef, p.numinstances)

                for i in 1:numinstances
                    if p.initial == "cold"
                        U[i] = identity_gauges(NX, NY, NZ, NT, p.beta, type_of_gaction=GA)
                    elseif p.initial == "hot"
                        U[i] = random_gauges(NX, NY, NZ, NT, p.beta, type_of_gaction=GA)
                    else
                        error("Only \"hot\" or \"cold\" initial config valid.")
                    end

                    Bias[i] = BiasPotential(p, U[1]; instance=i-1, has_fp=fp)
                    write_to_file(Bias[i]; force=true)
                end

            else
                numinstances = 1

                if p.initial == "cold"
                    U = identity_gauges(NX, NY, NZ, NT, p.beta, type_of_gaction=GA)
                elseif p.initial == "hot"
                    U = random_gauges(NX, NY, NZ, NT, p.beta, type_of_gaction=GA)
                else
                    error("Only \"hot\" or \"cold\" initial config valid.")
                end

                Bias = BiasPotential(p, U; has_fp=fp)
                write_to_file(Bias; force=fp)
            end
        else
            tempering_enabled = p.tempering_enabled
            @assert tempering_enabled == false "tempering can only be enabled with MetaD"
            numinstances = 1
            Bias = nothing

            if p.initial == "cold"
                U = identity_gauges(NX, NY, NZ, NT, p.beta, type_of_gaction=GA)
            elseif p.initial == "hot"
                U = random_gauges(NX, NY, NZ, NT, p.beta, type_of_gaction=GA)
            end

        end

        if p.verboselevel == 1
            verbose_print = fp == true ? Verbose1(p.load_fp) : Verbose1()
        elseif p.verboselevel == 2
            verbose_print = fp == true ? Verbose2(p.load_fp) : Verbose2()
        elseif p.verboselevel == 3
            verbose_print = fp == true ? Verbose3(p.load_fp) : Verbose3()
        end

        return Univ{typeof(U), typeof(Bias), typeof(verbose_print)}(
            p.meta_enabled,
            tempering_enabled,
            U,
            Bias,
            numinstances,
            verbose_print,
        )
    end

    function get_cvsmearing_from_parameters(p::ParameterSet)
        if p.numsmears_for_cv == 0 || p.rhostout_for_cv == 0
            return NoSmearing
        elseif p.numsmears_for_cv > 0 && p.rhostout_for_cv > 0
            return StoutSmearing
        else
            error("numsmears_for_cv and rhostout_for_cv must be >0")
        end
    end
end

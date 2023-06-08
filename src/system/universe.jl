module UniverseModule
    using ..Utils
    using ..VerbosePrint

    import ..Gaugefields: AbstractGaugeAction, Gaugefield, Liefield
    import ..Gaugefields: DBW2GaugeAction, IwasakiGaugeAction, SymanzikTadGaugeAction
    import ..Gaugefields: SymanzikTreeGaugeAction, WilsonGaugeAction 
    import ..Gaugefields: identity_gauges, random_gauges
    import ..Gaugefields: Liefield
    import ..Metadynamics: BiasPotential
    import ..SystemParameters: Params

    struct Univ{TG}
        L::NTuple{4, Int64}
        NC::Int64
        meta_enabled::Bool
        tempering_enabled::Bool
        U::Vector{Gaugefield{TG}}
        Bias::Union{Vector{Nothing}, Vector{BiasPotential{TG}}}
        numinstances::Int64
        verbose_print::VerboseLevel
    end

    function Univ(p::Params)
        NX, NY, NZ, NT = p.L
        NC = 3

        if p.kind_of_gaction == "wilson"
            TG = WilsonGaugeAction
        elseif p.kind_of_gaction == "symanzik_tree"
            TG = SymanzikTreeGaugeAction
        elseif p.kind_of_gaction == "symanzik_tadpole"
            TG = SymanzikTadGaugeAction
        elseif p.kind_of_gaction == "iwasaki"
            TG = IwasakiGaugeAction
        elseif p.kind_of_gaction == "dbw2"
            TG = DBW2GaugeAction
        else
            error("Gauge action '$(kind_of_gaction)' not supported")
        end

        U = Vector{Gaugefield{TG}}(undef, 0)

        if p.meta_enabled
            Bias = Vector{BiasPotential{TG}}(undef, 0)
            tempering_enabled = p.tempering_enabled

            if tempering_enabled
                numinstances = p.numinstances

                for i in 1:numinstances
                    if p.initial == "cold"
                        push!(U, identity_gauges(
                            NX, NY, NZ, NT,
                            p.β,
                            type_of_gaction = TG,
                        ))
                    elseif p.initial == "hot"
                        push!(U, random_gauges(
                            NX, NY, NZ, NT,
                            p.β,
                            type_of_gaction = TG,
                        ))
                    else
                        error("Initial \"$(p.initial)\" is invalid")
                    end

                    push!(Bias, BiasPotential(p, U[1], instance = i))
                end
            else
                numinstances = 1

                if p.initial == "cold"
                    push!(U, identity_gauges(
                        NX, NY, NZ, NT,
                        p.β,
                        type_of_gaction = TG,
                    ))
                elseif p.initial == "hot"
                    push!(U, random_gauges(
                        NX, NY, NZ, NT,
                        p.β,
                        type_of_gaction = TG,
                    ))
                end

                push!(Bias, BiasPotential(p, U[1]))
            end
        else
            Bias = Vector{Nothing}(undef, 1)
            tempering_enabled = false
            numinstances = 1

            if p.initial == "cold"
                push!(U, identity_gauges(
                    NX, NY, NZ, NT,
                    p.β,
                    type_of_gaction = TG,
                ))
            elseif p.initial == "hot"
                push!(U, random_gauges(
                    NX, NY, NZ, NT,
                    p.β,
                    type_of_gaction = TG,
                ))
            end

        end

        if p.verboselevel == 1
            verbose_print = Verbose1(p.load_fp)
        elseif p.verboselevel == 2
            verbose_print = Verbose2(p.load_fp)
        elseif p.verboselevel == 3
            verbose_print = Verbose3(p.load_fp)
        end

        return Univ{TG}(
            p.L,
            NC,
            p.meta_enabled,
            tempering_enabled,
            U,
            Bias,
            numinstances,
            verbose_print,
        )
    end

end
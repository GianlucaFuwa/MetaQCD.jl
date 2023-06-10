module UniverseModule
    using ..Utils
    using ..VerbosePrint

    import ..Gaugefields: Gaugefield, Liefield
    import ..Gaugefields: AbstractGaugeAction, DBW2GaugeAction, IwasakiGaugeAction,
        SymanzikTadGaugeAction, SymanzikTreeGaugeAction, WilsonGaugeAction 
    import ..Gaugefields: identity_gauges, random_gauges
    import ..Metadynamics: BiasPotential, MetaDisabled, MetaEnabled
    import ..SystemParameters: Params

    struct Univ{TG, TM, TV}
        L::NTuple{4, Int64}
        NC::Int64
        meta_enabled::Bool
        tempering_enabled::Bool
        U::Vector{Gaugefield{TG}}
        Bias::Union{Vector{Nothing}, Vector{BiasPotential{TG}}}
        numinstances::Int64
        verbose_print::TV
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

        if p.meta_enabled
            TM = MetaEnabled
            U = Vector{Gaugefield{TG}}(undef, p.numinstances)
            Bias = Vector{BiasPotential{TG}}(undef, p.numinstances)
            tempering_enabled = p.tempering_enabled

            if tempering_enabled
                numinstances = p.numinstances

                for i in 1:numinstances
                    if p.initial == "cold"
                        U[i] = identity_gauges(
                            NX, NY, NZ, NT,
                            p.β,
                            type_of_gaction = TG,
                        )
                    elseif p.initial == "hot"
                        U[i] = random_gauges(
                            NX, NY, NZ, NT,
                            p.β,
                            type_of_gaction = TG,
                        )
                    else
                        error("Initial \"$(p.initial)\" is invalid")
                    end

                    push!(Bias, BiasPotential(p, U[1], instance = i))
                end
            else
                numinstances = 1
                U = Vector{Gaugefield{TG}}(undef, 1)

                if p.initial == "cold"
                    U[1] = identity_gauges(
                        NX, NY, NZ, NT,
                        p.β,
                        type_of_gaction = TG,
                    )
                elseif p.initial == "hot"
                    U[1] = random_gauges(
                        NX, NY, NZ, NT,
                        p.β,
                        type_of_gaction = TG,
                    )
                end

                Bias[1] = BiasPotential(p, U[1])
            end
        else
            TM = MetaDisabled
            U = Vector{Gaugefield{TG}}(undef, 1)
            Bias = Vector{Nothing}(undef, 1)
            tempering_enabled = false
            numinstances = 1

            if p.initial == "cold"
                U[1] = identity_gauges(
                    NX, NY, NZ, NT,
                    p.β,
                    type_of_gaction = TG,
                )
            elseif p.initial == "hot"
                U[1] = random_gauges(
                    NX, NY, NZ, NT,
                    p.β,
                    type_of_gaction = TG,
                )
            end

        end

        if p.verboselevel == 1
            verbose_print = Verbose1(p.load_fp)
        elseif p.verboselevel == 2
            verbose_print = Verbose2(p.load_fp)
        elseif p.verboselevel == 3
            verbose_print = Verbose3(p.load_fp)
        end

        return Univ{TG, TM, typeof(verbose_print)}(
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
module UniverseModule
    using ..Utils
    using ..VerbosePrint

    import ..Gaugefields: Gaugefield, Liefield
    import ..Gaugefields: AbstractGaugeAction, DBW2GaugeAction, IwasakiGaugeAction,
        SymanzikTadGaugeAction, SymanzikTreeGaugeAction, WilsonGaugeAction
    import ..Gaugefields: identity_gauges, random_gauges
    import ..Metadynamics: BiasPotential, MetaDisabled, MetaEnabled
    import ..SystemParameters: Params

    struct Univ{TG, TB, TM, TV}
        meta_enabled::Bool
        tempering_enabled::Bool
        U::TG
        Bias::TB
        numinstances::Int64
        verbose_print::TV
    end

    function Univ(p::Params; use_mpi = false, fp = true)
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
            TM = MetaEnabled
            tempering_enabled = p.tempering_enabled

            if tempering_enabled && use_mpi == false
                numinstances = p.numinstances
                U = Vector{Gaugefield{GA}}(undef, p.numinstances)
                Bias = Vector{BiasPotential{Gaugefield{GA}}}(undef, p.numinstances)

                for i in 1:numinstances
                    if p.initial == "cold"
                        U[i] = identity_gauges(
                            NX, NY, NZ, NT,
                            p.β,
                            type_of_gaction = GA,
                        )
                    elseif p.initial == "hot"
                        U[i] = random_gauges(
                            NX, NY, NZ, NT,
                            p.β,
                            type_of_gaction = GA,
                        )
                    else
                        error("Initial \"$(p.initial)\" is invalid")
                    end

                    Bias[i] = BiasPotential(p, U[1], instance = i - 1, has_fp = fp)
                end

            else
                numinstances = 1

                if p.initial == "cold"
                    U = identity_gauges(
                        NX, NY, NZ, NT,
                        p.β,
                        type_of_gaction = GA,
                    )
                elseif p.initial == "hot"
                    U = random_gauges(
                        NX, NY, NZ, NT,
                        p.β,
                        type_of_gaction = GA,
                    )
                end

                Bias = BiasPotential(p, U; has_fp = fp)
            end
        else
            tempering_enabled = p.tempering_enabled
            @assert tempering_enabled == false "tempering can only be enabled with MetaD"
            TM = MetaDisabled
            numinstances = 1
            Bias = nothing

            if p.initial == "cold"
                U = identity_gauges(
                    NX, NY, NZ, NT,
                    p.β,
                    type_of_gaction = GA,
                )
            elseif p.initial == "hot"
                U = random_gauges(
                    NX, NY, NZ, NT,
                    p.β,
                    type_of_gaction = GA,
                )
            end

        end

        if p.verboselevel == 1
            verbose_print = fp == true ? Verbose1(p.load_fp) : Verbose1()
        elseif p.verboselevel == 2
            verbose_print = fp == true ? Verbose2(p.load_fp) : Verbose1()
        elseif p.verboselevel == 3
            verbose_print = fp == true ? Verbose3(p.load_fp) : Verbose1()
        end

        return Univ{typeof(U), typeof(Bias), TM, typeof(verbose_print)}(
            p.meta_enabled,
            tempering_enabled,
            U,
            Bias,
            numinstances,
            verbose_print,
        )
    end

end

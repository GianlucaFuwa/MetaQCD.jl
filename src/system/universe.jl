module UniverseModule
    using ..Utils
    using ..VerbosePrint

    import ..Gaugefields: Gaugefield, identity_gauges, random_gauges
    import ..Gaugefields: Liefield
    import ..Metadynamics: BiasPotential
    import ..SystemParameters: Params

    struct Univ
        L::NTuple{4, Int64}
        NC::Int64
        meta_enabled::Bool
        tempering_enabled::Bool
        U::Vector{Gaugefield}
        Bias::Union{Nothing, Vector{BiasPotential}}
        numinstances::Int64
        verbose_print::VerboseLevel
    end

    function Univ(p::Params)
        L = p.L
        NC = 3
        meta_enabled = p.meta_enabled
        tempering_enabled = p.tempering_enabled
        U = Vector{Gaugefield}(undef, 0)

        if meta_enabled
            Bias = Vector{BiasPotential}(undef, 0)
            if tempering_enabled
                numinstances = p.numinstances
                for i in 1:numinstances

                    if p.initial == "cold"
                        push!(U, identity_gauges(
                            L[1], L[2], L[3], L[4],
                            p.β,
                            kind_of_gaction = p.kind_of_gaction,
                        ))
                    else
                        push!(U,
                        random_gauges(L[1], L[2], L[3], L[4],
                        p.β,
                        kind_of_gaction = p.kind_of_gaction,
                        rng = p.randomseeds[i],
                    ))
                    end

                    push!(Bias, BiasPotential(p, i))
                end
            else
                numinstances = 1

                if p.initial == "cold"
                    push!(U, identity_gauges(
                        L[1], L[2], L[3], L[4],
                        p.β,
                        kind_of_gaction = p.kind_of_gaction,
                    ))
                else
                    push!(U, random_gauges(
                        L[1], L[2], L[3], L[4],
                        p.β,
                        kind_of_gaction = p.kind_of_gaction,
                        rng = p.randomseeds[1],
                    ))
                end

                push!(Bias, BiasPotential(p))
            end
        else
            Bias = nothing
            tempering_enabled = false
            numinstances = 1

            if p.initial == "cold"
                push!(U, identity_gauges(
                    L[1], L[2], L[3], L[4],
                    p.β,
                    kind_of_gaction = p.kind_of_gaction,
                ))
            else
                push!(U, random_gauges(
                    L[1], L[2], L[3], L[4],
                    p.β,
                    kind_of_gaction = p.kind_of_gaction,
                    rng = p.randomseeds[1],
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

        return Univ(
            L,
            NC,
            meta_enabled,
            tempering_enabled,
            U,
            Bias,
            numinstances,
            verbose_print,
        )
    end

end
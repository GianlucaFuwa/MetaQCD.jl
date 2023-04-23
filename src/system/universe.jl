module Universe_module
    import ..System_parameters: Params
    import ..Verbose_print: Verbose_level, Verbose_1, Verbose_2, Verbose_3
    import ..Gaugefields: Gaugefield, IdentityGauges, RandomGauges
    import ..Liefields: Liefield
    import ..Metadynamics: Bias_potential

    struct Univ
        L::NTuple{4,Int64}
        NC::Int64
        meta_enabled::Bool
        tempering_enabled::Bool
        U::Vector{Gaugefield}
        Bias::Union{Nothing,Vector{Bias_potential}}
        numinstances::Int64
        verbose_print::Verbose_level
    end

    function Univ(p::Params)
        L = p.L
        NC = 3
        meta_enabled = p.meta_enabled
        tempering_enabled = p.tempering_enabled
        U = Vector{Gaugefield}(undef, 0)
        if meta_enabled
            Bias = Vector{Bias_potential}(undef, 0)
            if tempering_enabled
                numinstances = p.numinstances
                for i = 1:numinstances
                    if p.initial == "cold"
                        push!(U, IdentityGauges(L[1], L[2], L[3], L[4], p.β, kind_of_gaction=p.kind_of_gaction))
                    else
                        push!(U, RandomGauges(L[1], L[2], L[3], L[4], p.β, kind_of_gaction=p.kind_of_gaction, rng=p.randomseeds[i]))
                    end
                    push!(Bias, Bias_potential(p, i))
                end
            else
                numinstances = 1
                if p.initial == "cold"
                    push!(U, IdentityGauges(L[1], L[2], L[3], L[4], p.β, kind_of_gaction=p.kind_of_gaction))
                else
                    push!(U, RandomGauges(L[1], L[2], L[3], L[4], p.β, kind_of_gaction=p.kind_of_gaction, rng=p.randomseeds[1]))
                end
                push!(Bias, Bias_potential(p))
            end
        else
            Bias = nothing
            tempering_enabled = false
            numinstances = 1
            if p.initial == "cold"
                push!(U, IdentityGauges(L[1], L[2], L[3], L[4], p.β, kind_of_gaction=p.kind_of_gaction))
            else
                push!(U, RandomGauges(L[1], L[2], L[3], L[4], p.β, kind_of_gaction=p.kind_of_gaction, rng=p.randomseeds[1]))
            end
        end

        if p.verboselevel == 1
            verbose_print = Verbose_1(p.load_fp)
        elseif p.verboselevel == 2
            verbose_print = Verbose_2(p.load_fp)
        elseif p.verboselevel == 3
            verbose_print = Verbose_3(p.load_fp)
        end

        return Univ(L,
            NC,
            meta_enabled,
            tempering_enabled,
            U,
            Bias,
            numinstances,
            verbose_print)
    end

end





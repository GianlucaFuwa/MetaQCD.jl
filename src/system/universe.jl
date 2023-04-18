module Universe_module
    import ..System_parameters: Params
    import ..Verbose_print: Verbose_level,Verbose_1,Verbose_2,Verbose_3
    import ..Gaugefields: Gaugefield
    import ..Liefields: Liefield
    import ..Metadynamics: Bias_potential

    struct Univ
        L::NTuple{4,Int64}
        NC::Int64
        meta_enabled::Bool
        tempering_enabled::Bool
        U::Union{Gaugefield,Vector{Gaugefield}}
        P::Union{Nothing,Liefield,Vector{Liefield}}
        Bias::Union{Nothing,Bias_potential,Vector{Bias_potential}}
        verbose_print::Verbose_level
    end

    function Univ(p::Params)
        L = p.L
        NC = 3
        meta_enabled = p.meta_enabled
        
        if meta_enabled
            if tempering_enabled
                U = Vector{Gaugefield}(undef, 0)
                P = Vector{Liefield}(undef, 0)
                Bias = Vector{Bias_potential}(undef, 0)
                for i = 1:p.numinstances
                    push!(U, Gaugefield(p))
                    push!(P, Liefield(U[i]))
                    push!(Bias, Bias_potential(p))
                    push!(updatemethod, p.updatemethod[i])
                end
            else
                U = Gaugefield(p)
                P = Liefield(U)
                Bias = Bias_potential(p)
            end
        else
            tempering_enabled = false
            U = Gaugefield(p)
            P = Liefield(U)
        end
        if p.verboselevel == 1
            verbose_print = Verbose_1(p.logfile)
        elseif p.verboselevel == 2
            verbose_print = Verbose_2(p.logfile)
        elseif p.verboselevel == 3
            verbose_print = Verbose_3(p.logfile)
        end

        return Univ(L,
            NC,
            meta_enabled,
            tempering_enabled,
            U,
            P,
            Bias,
            verbose_print)
    end

end





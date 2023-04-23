module Tempering_module
    using Random

    import ..System_parameters: Params
    import ..Gaugefields: Gaugefield,get_CV,get_Sg,set_CV!,set_Sg!,substitute_U!
    import ..Metadynamics: Bias_potential,update_bias!,DeltaV,is_static

    function temper!(g1::Vector{T},g2::Vector{T},bias1::Bias_potential,bias2::Bias_potential,rng::Xoshiro=Xoshiro()) where {T<:Gaugefield}
        cv1 = get_CV(g1)
        cv2 = get_CV(g2)

        ΔV1 = DeltaV(bias1,cv1,cv2)
        ΔV2 = DeltaV(bias2,cv2,cv1)

        accept_swap = rand(rng) ≤ exp(ΔV1 + ΔV2)
        if accept_swap
            swap_gauges!(g1,g2)
            is_static(bias1) ? nothing : update_bias!(bias1,cv2)
            is_static(bias2) ? nothing : update_bias!(bias2,cv1)
        end
        return accept_swap
    end

    function swap_gauges!(g1::Vector{T},g2::Vector{T}) where {T<:Gaugefield}
        tmp = deepcopy(g1)
        substitute_U!(g1,g2)
        substitute_U!(g2,tmp)

        set_CV!(g1,get_CV(g2))
        set_CV!(g2,get_CV(tmp))
    
        set_Sg!(g1,get_Sg(g2))
        set_Sg!(g2,get_Sg(tmp))
        return nothing
    end

end




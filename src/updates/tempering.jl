module TemperingModule
    using Random

    import ..Gaugefields: Gaugefield, swap_U!
    import ..Metadynamics: BiasPotential, update_bias!
    import ..SystemParameters: Params

    function temper!(
        U1::Gaugefield,
        U2::Gaugefield,
        bias1::BiasPotential,
        bias2::BiasPotential,
        rng,
    )
        cv1 = U1.CV
        cv2 = U2.CV

        ΔV1 = DeltaV(bias1, cv1, cv2)
        ΔV2 = DeltaV(bias2, cv2, cv1)

        accept_swap = rand(rng) ≤ exp(ΔV1 + ΔV2)

        if accept_swap
            swap_U!(U1, U2)
            is_static(bias1) ? nothing : update_bias!(bias1, cv2)
            is_static(bias2) ? nothing : update_bias!(bias2, cv1)
        end

        return accept_swap
    end

end
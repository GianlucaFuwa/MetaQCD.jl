module TemperingModule
    import ..Gaugefields: Gaugefield, swap_U!
    import ..Metadynamics: BiasPotential, update_bias!

    function temper!(
        U1::Gaugefield,
        U2::Gaugefield,
        bias1::BiasPotential,
        bias2::BiasPotential,
    )
        cv1 = U1.CV
        cv2 = U2.CV

        ΔV1 = bias1(cv2) - bias1(cv1)
        ΔV2 = bias2(cv1) - bias2(cv2)

        accept_swap = rand() ≤ exp(ΔV1 + ΔV2)

        if accept_swap
            swap_U!(U1, U2)
            is_static(bias1) ? nothing : update_bias!(bias1, cv2)
            is_static(bias2) ? nothing : update_bias!(bias2, cv1)
        end

        return accept_swap
    end

end
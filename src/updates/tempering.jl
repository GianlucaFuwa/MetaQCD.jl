function temper!(U1, U2, bias1, bias2, itrj, verbose::VerboseLevel)
    cv1 = U1.CV
    cv2 = U2.CV
    ΔV1 = bias1(cv2) - bias1(cv1)
    ΔV2 = bias2(cv1) - bias2(cv2)

    println_verbose2(verbose, "ΔV1 = $(ΔV1)", "\n", "ΔV2 = $(ΔV2)")

    accept_swap = rand() ≤ exp(ΔV1 + ΔV2)
    if accept_swap
        swap_U!(U1, U2)
        update_bias!(bias1, cv2, itrj, true)
        update_bias!(bias2, cv1, itrj, true)
    end

    return accept_swap
end

function swap_U!(a, b)
    a_Sg_tmp = a.Sg
    a_CV_tmp = a.CV

    a.Sg = b.Sg
    a.CV = b.CV
    b.Sg = a_Sg_tmp
    b.CV = a_CV_tmp

    @batch for site in eachindex(a)
        for μ in 1:4
            a_tmp = a[μ][site]
            a[μ][site] = b[μ][site]
            b[μ][site] = a_tmp
        end
    end

    return nothing
end

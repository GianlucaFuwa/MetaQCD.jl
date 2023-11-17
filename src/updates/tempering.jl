function temper!(
    U::Vector{TG}, bias::Vector{TB}, numaccepts_temper, swap_every, itrj, verbose;
    recalc = false,
) where {TG<:Gaugefield, TB<:Bias}
    itrj%swap_every!=0 && return nothing
    numinstances = length(U)
    recalc && recalc_CV!(U[1], bias[1])

    for i in numinstances:-1:2
        U1 = U[i]
        U2 = U[i-1]
        bias1 = bias[i]
        bias2 = bias[i-1]
        cv1 = U1.CV
        cv2 = U2.CV
        ΔV1 = bias1(cv2) - bias1(cv1)
        ΔV2 = bias2(cv1) - bias2(cv2)
        acc_prob = exp(-ΔV1-ΔV2)
        println_verbose1(verbose, ">> ΔV$(i) = $(ΔV1)", "\t", "ΔV$(i-1) = $(ΔV2)")

        if rand() ≤ acc_prob
            println("# swap accepted")
            numaccepts_temper[i-1] += 1
            swap_U!(U1, U2)
            update_bias!(bias1, cv2, itrj, true)
            update_bias!(bias2, cv1, itrj, true)
        else
            println("# swap rejected")
        end

        println_verbose1(
            verbose,
            ">> Swap Acceptance [$i ⇔  $(i-1)]:\t",
            "$(100numaccepts_temper[i-1] / (itrj/swap_every)) %"
        )
    end

    return nothing
end

function swap_U!(a::Gaugefield, b::Gaugefield)
    a_Sg_tmp = deepcopy(a.Sg)
    a_CV_tmp = deepcopy(a.CV)

    a.Sg = b.Sg
    a.CV = b.CV
    b.Sg = a_Sg_tmp
    b.CV = a_CV_tmp

    @batch per=thread for site in eachindex(a)
        for μ in 1:4
            a_tmp = a[μ][site]
            a[μ][site] = b[μ][site]
            b[μ][site] = a_tmp
        end
    end

    return nothing
end

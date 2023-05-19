function recalc_gauge_action!(U::Gaugefield)
    U.Sg = calc_gauge_action(U)
    return nothing
end

function calc_gauge_action(U::Gaugefield)
    if U.kind_of_gaction == "wilson"
        Sg = gauge_action_wilson(U)
    elseif U.kind_of_gaction == "symanzik"
        Sg = gauge_action_symanzik(U)
    elseif U.kind_of_gaction == "iwasaki"
        Sg = gauge_action_iwasaki(U)
    elseif U.kind_of_gaction == "dbw2"
        Sg = gauge_action_dbw2(U)
    else
        error("type_of_gaction $(U.kind_of_gaction) is not supported!")
    end
    
    return Sg
end

function gauge_action_wilson(U::Gaugefield)
    P = plaquette_trace_sum(U)
    return U.β * (6 * U.NV - 1/3 * P)
end

function gauge_action_symanzik(U::Gaugefield)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    u0sq = sqrt(1/3 * P)
    Sg_plaq = 6 * U.NV - 1/3 * P
    Sg_rect = 12 * U.NV - 1/3 * R
    Sg_symanzik = U.β * ((1 + 8/12) * Sg_plaq - 1/12u0sq * Sg_rect) 
    return Sg_symanzik
end

function gauge_action_iwasaki(U::Gaugefield)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    Sg_plaq = 6 * U.NV - 1/3 * P
    Sg_rect = 12 * U.NV - 1/3 * R
    Sg_iwasaki = U.β * ((1 + 8*0.331) * Sg_plaq - 0.331 * Sg_rect) 
    return Sg_iwasaki
end

function gauge_action_dbw2(U::Gaugefield)
    P = plaquette_trace_sum(U)
    R = rect_trace_sum(U)
    Sg_plaq = 6 * U.NV - 1/3 * P
    Sg_rect = 12 * U.NV - 1/3 * R
    Sg_dbw2 = U.β * ((1 + 8*1.409) * Sg_plaq - 1.409 * Sg_rect) 
    return Sg_dbw2
end
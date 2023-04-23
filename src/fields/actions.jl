function recalc_GaugeAction!(U::T) where {T<:Gaugefield}
    U.Sg = calc_GaugeAction(U)
    return nothing
end

function calc_GaugeAction(U::T) where {T<:Gaugefield}
    if U.kind_of_gaction == "Wilson"
        Sg = GaugeAction_Wilson(U)
    elseif U.kind_of_gaction == "Symanzik"
        Sg = GaugeAction_Symanzik(U)
    elseif U.kind_of_gaction == "Iwasaki"
        Sg = GaugeAction_Iwasaki(U)
    elseif U.kind_of_gaction == "DBW2"
        Sg = GaugeAction_DBW2(U)
    else
        error("type_of_gaction $(U.kind_of_gaction) is not supported!")
    end
    return Sg
end

function GaugeAction_Wilson(U::T) where {T<:Gaugefield}
    P = plaquette_tracedsum(U)
    return U.β * (6 * U.NV - 1/3 * P)
end

function GaugeAction_Symanzik(U::T) where {T <: Gaugefield}
    P = plaquette_tracedsum(U)
    R = rect_tracedsum(U)
    u0sq = sqrt(1/3 * P)
    #u0sq = 1.0
    Sg_plaq = 6 * U.NV - 1/3 * P
    Sg_rect = 12 * U.NV - 1/3 * R
    Sg_symanzik = U.β * ( (1 + 8/12) * Sg_plaq - 1/12u0sq * Sg_rect) 
    return Sg_symanzik
end

function GaugeAction_Iwasaki(U::T) where {T<:Gaugefield}
    P = plaquette_tracedsum(U)
    R = rect_tracedsum(U)
    Sg_plaq = 6 * U.NV - 1/3 * P
    Sg_rect = 12 * U.NV - 1/3 * R
    Sg_iwasaki = U.β * ( (1 + 8*0.331) * Sg_plaq - 0.331 * Sg_rect ) 
    return Sg_iwasaki
end

function GaugeAction_DBW2(U::T) where {T<:Gaugefield}
    P = plaquette_tracedsum(U)
    R = rect_tracedsum(U)
    Sg_plaq = 6 * U.NV - 1/3 * P
    Sg_rect = 12 * U.NV - 1/3 * R
    Sg_dbw2 = U.β * ( (1 + 8*1.409) * Sg_plaq - 1.409 * Sg_rect ) 
    return Sg_dbw2
end

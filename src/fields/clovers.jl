function clover_square(U::T, μ, ν, site::Site_coords; L = 1) where {T <: Gaugefield}
    @assert L > 0 "Square side-lengths for clover have to be positive"

    topright    = wilsonloop_top_right(U, μ, ν, site, L, L)
    topleft     = wilsonloop_top_left(U, μ, ν, site, L, L)
    bottomleft  = wilsonloop_bottom_left(U, μ, ν, site, L, L)
    bottomright = wilsonloop_bottom_right(U, μ, ν, site, L, L)

    return topright + topleft + bottomleft + bottomright
end

function clover_rect(U::T, μ, ν, site::Site_coords; L1 = 1, L2 = 2) where {T <: Gaugefield}
    @assert L1 > 0 && L2 > 0 "Rectangle side-lengths for clover have to be positive"

    topright1x2    = wilsonloop_top_right(U, μ, ν, site, L1, L2)
    topleft1x2     = wilsonloop_top_right(U, μ, ν, site, L1, L2)
    bottomleft1x2  = wilsonloop_top_right(U, μ, ν, site, L1, L2)
    bottomright1x2 = wilsonloop_top_right(U, μ, ν, site, L1, L2)

    topright2x1    = wilsonloop_top_right(U, μ, ν, site, L2, L1)
    topleft2x1     = wilsonloop_top_right(U, μ, ν, site, L2, L1)
    bottomleft2x1  = wilsonloop_top_right(U, μ, ν, site, L2, L1)
    bottomright2x1 = wilsonloop_top_right(U, μ, ν, site, L2, L1)

    return topright1x2 + topleft1x2 + bottomleft1x2 + bottomright1x2 + 
           topright2x1 + topleft2x1 + bottomleft2x1 + bottomright2x1
end

function clover_derivative(U::T, μ, ν, site::Site_coords) where {T <: Gaugefield}
end
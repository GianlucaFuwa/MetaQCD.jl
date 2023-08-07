function clover_square(U, μ, ν, site, L)
    clover = @SMatrix zeros(ComplexF64, 3, 3)

    clover += wilsonloop_top_right(U, μ, ν, site, L, L)
    clover += wilsonloop_top_left(U, μ, ν, site, L, L)
    clover += wilsonloop_bottom_left(U, μ, ν, site, L, L)
    clover += wilsonloop_bottom_right(U, μ, ν, site, L, L)

    return clover
end

function clover_rect(U, μ, ν, site, L1, L2)
    clover = @SMatrix zeros(ComplexF64, 3, 3)

    clover += wilsonloop_top_right(U, μ, ν, site, L1, L2)
    clover += wilsonloop_top_left(U, μ, ν, site, L1, L2)
    clover += wilsonloop_bottom_left(U, μ, ν, site, L1, L2)
    clover += wilsonloop_bottom_right(U, μ, ν, site, L1, L2)

    clover += wilsonloop_top_right(U, μ, ν, site, L2, L1)
    clover += wilsonloop_top_left(U, μ, ν, site, L2, L1)
    clover += wilsonloop_bottom_left(U, μ, ν, site, L2, L1)
    clover += wilsonloop_bottom_right(U, μ, ν, site, L2, L1)

    return clover
end

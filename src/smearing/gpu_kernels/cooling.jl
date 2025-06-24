function cool!(Uflow::Gaugefield{B}) where {B}
    GA = WilsonGaugeAction()
    # TODO: can hide
    update_halo!(Uflow)
    @latmap(eachindex(Uflow), cool_gpu!, GA)
    return nothing
end

@kernel cpu=false function cool_gpu!(Uflow, GA, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    @unroll for μ in 1i32:4i32
        old_link = Uflow[μ, site]
        A_adj = staple(GA, Uflow, μ, site)'
        @inbounds Uflow[μ, site] = proj_onto_SU3(cooling_SU3(old_link, A_adj))
    end
end

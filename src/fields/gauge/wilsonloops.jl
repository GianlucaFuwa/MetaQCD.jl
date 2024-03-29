function wilsonloop(U, μ, ν, site, Lμ, Lν)
    right = sign(Lμ)==1
    top = sign(Lν)==1

    if right && top
        return wilsonloop_top_right(U, μ, ν, site, Lμ, Lν)
    elseif !right && top
        return wilsonloop_top_left(U, μ, ν, site, Lμ, Lν)
    elseif right && !top
        return wilsonloop_bottom_right(U, μ, ν, site, Lμ, Lν)
    else
        return wilsonloop_bottom_left(U, μ, ν, site, Lμ, Lν)
    end

end

function wilsonloop_top_right(U, μ, ν, site, Lμ, Lν)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    wil = eye3

    for _ in 1:Lμ
        wil = cmatmul_oo(wil, U[μ][site])
        site = move(site, μ, 1, Nμ)
    end

    for _ in 1:Lν
        wil = cmatmul_oo(wil, U[ν][site])
        site = move(site, ν, 1, Nν)
    end

    for _ in 1:Lμ
        site = move(site, μ, -1, Nμ)
        wil = cmatmul_od(wil, U[μ][site])
    end

    for _ in 1:Lν
        site = move(site, ν, -1, Nν)
        wil = cmatmul_od(wil, U[ν][site])
    end

    return wil
end

function wilsonloop_bottom_left(U, μ, ν, site, Lμ, Lν)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    wil = eye3

    for _ in 1:Lμ
        site = move(site, μ, -1, Nμ)
        wil = cmatmul_od(wil, U[μ][site])
    end

    for _ in 1:Lν
        site = move(site, ν, -1, Nν)
        wil = cmatmul_od(wil, U[ν][site])
    end

    for _ in 1:Lμ
        wil = cmatmul_oo(wil, U[μ][site])
        site = move(site, μ, 1, Nμ)
    end

    for _ in 1:Lν
        wil = cmatmul_oo(wil, U[ν][site])
        site = move(site, ν, 1, Nν)
    end

    return wil
end

function wilsonloop_top_left(U, μ, ν, site, Lμ, Lν)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    wil = eye3

    for _ in 1:Lν
        wil = cmatmul_oo(wil, U[ν][site])
        site = move(site, ν, 1, Nν)
    end

    for _ in 1:Lμ
        site = move(site, μ, -1, Nμ)
        wil = cmatmul_od(wil, U[μ][site])
    end

    for _ in 1:Lν
        site = move(site, ν, -1, Nν)
        wil = cmatmul_od(wil, U[ν][site])
    end

    for _ in 1:Lμ
        wil = cmatmul_oo(wil, U[μ][site])
        site = move(site, μ, 1, Nμ)
    end

    return wil
end

function wilsonloop_bottom_right(U, μ, ν, site, Lμ, Lν)
    Nμ = size(U)[μ]
    Nν = size(U)[ν]
    wil = eye3

    for _ in 1:Lν
        site = move(site, ν, -1, Nν)
        wil = cmatmul_od(wil, U[ν][site])
    end

    for _ in 1:Lμ
        wil = cmatmul_oo(wil, U[μ][site])
        site = move(site, μ, 1, Nμ)
    end

    for _ in 1:Lν
        wil = cmatmul_oo(wil, U[ν][site])
        site = move(site, ν, 1, Nν)
    end

    for _ in 1:Lμ
        site = move(site, μ, -1, Nμ)
        wil = cmatmul_od(wil, U[μ][site])
    end

    return wil
end

function wilsonloop(U, Lμ, Lν)
    out = zeros(Float64, 8nthreads())

    @batch per=thread for site in eachindex(U)
        for μ in 1:3
            for ν in μ+1:4
                out[8threadid()] += real(tr(wilsonloop(U, μ, ν, site, Lμ, Lν)))
            end
        end
    end

    return sum(out)
end

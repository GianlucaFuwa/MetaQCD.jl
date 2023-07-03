function saveU_jld(U, filename)
    save(filename, "U", U)
    return nothing
end

function loadU_jld!(U, filename)
    Unew = load(filename, "U")

    @assert Unew.NX == U.NX
    @assert Unew.NY == U.NY
    @assert Unew.NZ == U.NZ
    @assert Unew.NT == U.NT

    for μ in 1:4
        for ii in eachindex(U)
            U[μ][ii] = Unew[μ][ii]
        end
    end

    return nothing
end

function loadU_jld(filename)
    Unew = load(filename, "U")
    return Unew
end

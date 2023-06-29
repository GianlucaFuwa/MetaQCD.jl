module JLD2Format
    using JLD2
    using Polyester

    import ..Gaugefields: Gaugefield

    function save_U(filename, U::T) where {T <: Gaugefield}
        save(filename, "U", U)
        return nothing
    end

    function load_U!(filename, U::T) where {T <: Gaugefield}
        Unew = load(filename, "U")

        @assert Unew.NX == U.NX
        @assert Unew.NY == U.NY
        @assert Unew.NZ == U.NZ
        @assert Unew.NT == U.NT

        @batch for μ in 1:4
            for ii in eachindex(U)
                U[μ][ii] = Unew[μ][ii]
            end
        end

        return nothing
    end

    function load_U(filename)
        Unew = load(filename, "U")
        return Unew
    end

end

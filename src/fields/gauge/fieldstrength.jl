abstract type AbstractFieldstrength end

struct Plaquette <: AbstractFieldstrength end
struct Clover <: AbstractFieldstrength end
struct Improved <: AbstractFieldstrength end

function fieldstrength_eachsite!(F::Vector{Temporaryfield}, U, kind_of_fs::String)
    if kind_of_fs == "plaquette"
        fieldstrength_eachsite!(Plaquette(), F, U)
    elseif kind_of_fs == "clover"
        fieldstrength_eachsite!(Clover(), F, U)
    else
        error("kind of fieldstrength \"$(kind_of_fs)\" not supported")
    end

    return nothing
end

function fieldstrength_eachsite!(::Plaquette, F::Vector{Temporaryfield}, U)
    @batch for site in eachindex(U)
        C12 = plaquette(U, 1, 2, site)
        F[1][2][site] = im * traceless_antihermitian(C12)

        C13 = plaquette(U, 1, 3, site)
        F[1][3][site] = im * traceless_antihermitian(C13)

        C14 = plaquette(U, 1, 4, site)
        F[1][4][site] = im * traceless_antihermitian(C14)

        C23 = plaquette(U, 2, 3, site)
        F[2][3][site] = im * traceless_antihermitian(C23)

        C24 = plaquette(U, 2, 4, site)
        F[2][4][site] = im * traceless_antihermitian(C24)

        C34 = plaquette(U, 3, 4, site)
        F[3][4][site] = im * traceless_antihermitian(C34)
    end

    return nothing
end

function fieldstrength_eachsite!(::Clover, F::Vector{Temporaryfield}, U)
    @batch for site in eachindex(U)
        C12 = clover_square(U, 1, 2, site, 1)
        F[1][2][site] = im/4 * traceless_antihermitian(C12)

        C13 = clover_square(U, 1, 3, site, 1)
        F[1][3][site] = im/4 * traceless_antihermitian(C13)

        C14 = clover_square(U, 1, 4, site, 1)
        F[1][4][site] = im/4 * traceless_antihermitian(C14)

        C23 = clover_square(U, 2, 3, site, 1)
        F[2][3][site] = im/4 * traceless_antihermitian(C23)

        C24 = clover_square(U, 2, 4, site, 1)
        F[2][4][site] = im/4 * traceless_antihermitian(C24)

        C34 = clover_square(U, 3, 4, site, 1)
        F[3][4][site] = im/4 * traceless_antihermitian(C34)
    end

    return nothing
end

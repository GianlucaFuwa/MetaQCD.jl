"""
	Cooling(U::Gaugefield; numflow=1, measure_every=1)
	
Create a cooling struct that smears `numflow` times.
When used in `calc_measurements_flowed` observables are after every integer multiple of
`measure_every * rho` if `measure_every` is a positive integer and at
`measure_every[i] * tf for i in eachindex(measure_every)` if `measure_every` is a range.
"""
struct Cooling{TG} <: AbstractSmearing
    numflow::Int64
    tf::Float64
    measure_at::Vector{Int64}
    Uflow::TG
    function Cooling(U::TG; numflow=1, measure_every=1) where {TG}
        @level1("- Setting Cooling...")
        Uflow = similar(U)

        measure_at = if measure_every isa Int64
            range(measure_every, numflow; step=measure_every)
        elseif measure_every isa Vector{Int64}
            measure_every
        end

        tf = 1/3 # see arXiv:1708.00696

        @level1("|  NUMBER OF COOLING SWEEPS: $(numflow)")
        @level1("|  MEASURING ON COOLING NUMBERS: $(string(measure_at))")
        @level1("-\n")
        return new{TG}(numflow, tf, measure_at, Uflow)
    end
end

function Base.show(io::IO, ::MIME"text/plain", cool::Cooling)
    print(io, "Cooling(; numflow = $(cool.numflow))")
    return nothing
end

function Base.show(io::IO, cool::Cooling)
    print(io, "Cooling(; numflow = $(cool.numflow))")
    return nothing
end

@inline Base.:(==)(c1::T, c2::T) where {T<:Cooling} = (c1.numflow == c2.numflow)
Base.length(c::Cooling) = c.numflow

function flow!(cool::Cooling)
    Uflow = cool.Uflow
    cool!(Uflow)
    return nothing
end

function cool!(Uflow::Gaugefield{B,T,M}) where {B,T,M}
    GA = WilsonGaugeAction()

    parallelfor(eachindex(Uflow), B, Val(M), (), (Uflow,), (Uflow,)) do site, Uflow
        for μ in 1:4
            old_link = Uflow[μ, site]
            A_adj = staple(GA, Uflow, μ, site)'
            Uflow[μ, site] = proj_onto_SU3(cooling_SU3(old_link, A_adj))
        end
    end
end

function cooling_SU3(link, A_adj)
    subblock = make_submatrix_12(cmatmul_oo(link, A_adj))
    tmp = embed_into_SU3_12(cooling_SU2(subblock))
    link = cmatmul_oo(tmp, link)

    subblock = make_submatrix_13(cmatmul_oo(link, A_adj))
    tmp = embed_into_SU3_13(cooling_SU2(subblock))
    link = cmatmul_oo(tmp, link)

    subblock = make_submatrix_23(cmatmul_oo(link, A_adj))
    tmp = embed_into_SU3_23(cooling_SU2(subblock))
    link = cmatmul_oo(tmp, link)
    return link
end

@inline function cooling_SU2(A)
    a_norm = 1 / sqrt(det(A))
    return a_norm * A'
end

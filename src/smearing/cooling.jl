"""
	Cooling(U::Gaugefield; numlayers=1, measure_every=1)
	
Create a cooling struct that smears `numlayers` times.
When used in `calc_measurements_flowed` observables are after every integer multiple of
`measure_every * rho` if `measure_every` is a positive integer and at
`measure_every[i] * tf for i in eachindex(measure_every)` if `measure_every` is a range.
"""
struct Cooling{TG,TT,TC} <: AbstractSmearing
    numlayers::Int64
    measure_at::Vector{Int64}
    Ucool::TG
    function Cooling(U::TG; numlayers=1) where {TG}
        @level1("- Setting Cooling...")
        Ucool = similar(U)
        @level1("|  NUMBER OF COOLING SWEEPS: $(numflow)")
        @level1("|  MEASURING ON COOLING NUMBERS: $(string(measure_at))")
        @level1("-\n")
    end
end

function Base.show(io::IO, ::MIME"text/plain", cool::Cooling)
    print(io, "Cooling(; numlayers = $(cool.numlayers))")
    return nothing
end

function Base.show(io::IO, cool::Cooling)
    print(io, "Cooling(; numlayers = $(cool.numlayers))")
    return nothing
end

@inline Base.:(==)(c1::T, c2::T) where {T<:Cooling} = (c1.numlayers == c2.numlayers)
Base.length(c::Cooling) = c.numlayers

function flow!(method::Cooling, Uin)
    copy!(method.Ucool, Uin)
end

function cooling_SU2_kernel(A)
    a_norm = 1 / sqrt(det(A))
    return a_norm * A'
end

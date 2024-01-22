using Accessors
using Enzyme
using LinearAlgebra
using Random
using StaticArrays
import .EnzymeRules: forward, reverse, augmented_primal
using .EnzymeRules

const zero3 = @SArray [
    0.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 0.0+0.0im
]

const eye3 = @SArray [
    1.0+0.0im 0.0+0.0im 0.0+0.0im
    0.0+0.0im 1.0+0.0im 0.0+0.0im
    0.0+0.0im 0.0+0.0im 1.0+0.0im
]

Random.seed!(1206)

# U is filled with random special unitary matrices, but would make MWE too long
U = Array{SMatrix{3, 3, ComplexF64, 9}, 5}(undef, 4, 4, 4, 4, 4); fill!(U, eye3);
dU = similar(U); fill!(dU, zero3);
# U1 = initial_gauges("hot", 4, 4, 4, 4, 6.0);
# dU1 = Temporaryfield(U1);
# staples = Temporaryfield(U1);

# for μ in 1:4
#     view(U, μ, :, :, :, :) .= U1[μ]
# end

traceless_antihermitian(M::SMatrix{3,3,ComplexF64,9}) = 0.5*(M - M') - 1/6*tr(M - M')*eye3
# remultr(args...) = real(tr(*(args...)))
@generated function remultr(args::Vararg{T, N}) where {N,T}
    quote
        $(Expr(:meta, :inline))
        tmp = *(args...)
        real(tr(tmp))
    end
end
smove(s::CartesianIndex{4}, μ, steps, lim) = @set s[μ] = mod1(s[μ] + steps, lim)
@inline function plaquette(U, μ, ν, site)
    Nμ = size(U)[1+μ]
    Nν = size(U)[1+ν]
    siteμ⁺ = smove(site, μ, 1, Nμ)
    siteν⁺ = smove(site, ν, 1, Nν)
    return remultr(U[μ,site], U[ν,siteμ⁺], U[μ,siteν⁺], U[ν,site])
end

function plaquette_sum(U::Array{SMatrix{3,3,ComplexF64,9}, 5})
    p = 0.0

    for site in CartesianIndices(size(U)[2:end])
        for μ in 1:3
            for ν in μ+1:4
                p += plaquette(U, μ, ν, site)
            end
        end
    end

    return 6.0 * (6*prod(size(U)[2:end]) - 1/3*p)
end

### RULES FOR REMULTR ###
@inline function Base.circshift(shift::Integer, args::Vararg{T, N}) where {N,T}
    j = mod1(shift, N)
    ntuple(k -> args[k-j+ifelse(k>j,0,N)], Val(N))
end

function augmented_primal(config::ConfigWidth{1}, func::Const{typeof(remultr)},
    ::Type{<:Active}, args::Vararg{Active,N}) where {N}

    argvals = ntuple(i -> args[i].val, Val(N))
    if needs_primal(config)
        primal = func.val(argvals...)
    else
        primal = nothing
    end
    if overwritten(config)[3]
        tape = copy(argvals)
    else
        tape = nothing
    end

    return AugmentedReturn(primal, nothing, tape)
end

function EnzymeRules.reverse(config::ConfigWidth{1}, func::Const{typeof(remultr)},
    dret::Active, tape, args::Vararg{Active,N}) where {N}

    argvals = ntuple(i -> args[i].val, Val(N))
    dargs = ntuple(Val(N)) do i
        0.5traceless_antihermitian(*(circshift(i-1, argvals...)...))
    end
    return dargs
end

function Enzyme.gradient(::ReverseMode, f::typeof(remultr), args::Vararg{T,N}) where {N,T}
    annots = ntuple(i -> Active(args[i]), Val(N))
    der = autodiff(Reverse, f, Active, annots...)
    return der
end

### RULES FOR PLAQUETTE_SUM ###
function Enzyme.gradient!(::ReverseMode, dU::Array{SMatrix{3,3,ComplexF64,9},5},
    f::typeof(plaquette_sum), U::Array{SMatrix{3,3,ComplexF64,9},5})

    autodiff(Reverse, f, Active, DuplicatedNoNeed(U, dU))
    return nothing
end

# matrices = [gen_SU3_matrix(0.5) for _ in 1:4] # should be any random special unitary matrices
# dm = Enzyme.gradient(Reverse, remultr, matrices...) # works as wanted
Enzyme.gradient!(Reverse, dU, plaquette_sum, U) # AssertionError: !(overwritten[end])

const AnySpinorfield{B,T,M,A,ND} = Union{Spinorfield{B,T,M,A,ND},SpinorfieldEO{B,T,M,A,ND}}

function clear!(ϕ::AnySpinorfield{B}) where {B<:GPU}
    @latmap(eachindex(ϕ), clear_fermion_gpu!, ϕ)
end

@kernel cpu=false function clear_fermion_gpu!(ϕ, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    @inbounds ϕ[site] = zero(ϕ[site])
end

function Base.copy!(a::TF, b::TF) where {TF<:AnySpinorfield{<:GPU}}
    @latmap(eachindex(a, b), copy_fermion_gpu!, a, b)
    return nothing
end

@kernel cpu=false function copy_fermion_gpu!(a, @Const(b), bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    @inbounds a[site] = b[site]
end

function ones!(ϕ::AnySpinorfield{B}) where {B<:GPU}
    @latmap(eachindex(ϕ), ones_fermion_gpu!, ϕ)
    return nothing
end

@kernel cpu=false function ones_fermion_gpu!(ϕ, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    @inbounds ϕ[site] = fill(1, ϕ[site])
end

function set_source!(ϕ::AnySpinorfield{B,T}, site, a, μ) where {B<:GPU,T}
    NC = num_colors(ϕ)
    ND = num_dirac(ϕ)
    @assert μ ∈ 1:ND && a ∈ 1:NC
    @latmap(eachindex(ϕ), set_source_gpu!, ϕ, site, a, μ, NC, ND, T)
    return nothing
end

@kernel cpu=false function set_source_gpu!(ϕ, site, a, μ, NC, ND, ::Type{T}, bulk) where {T}
    iglobal = @index(Global, Cartesian)
    gsite = bulk[iglobal]
    if gsite == site
        vec_index = (μ - 1) * NC + a
        tup = ntuple(i -> i == vec_index ? one(Complex{T}) : zero(Complex{T}), Val(3ND))
        @inbounds ϕ[site] = SVector{3ND,Complex{T}}(tup)
    else
        @inbounds ϕ[site] = zero(ϕ[site])
    end
end

function gaussian_pseudofermions!(ϕ::AnySpinorfield{B,T,M,A,ND}) where {B<:GPU,T,M,A,ND}
    @latmap(eachindex(ϕ), gaussian_pseudofermions_gpu!, ϕ, Val(3ND), T)
    return nothing
end

@kernel cpu=false function gaussian_pseudofermions_gpu!(ϕ, ::Val{L}, ::Type{T}, bulk) where {L,T}
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    @inbounds ϕ[site] = randn(SVector{L,Complex{T}}) # σ = 0.5
end

function LinearAlgebra.mul!(ψ::TF, ϕ::TF, α) where {T,TF<:AnySpinorfield{<:GPU,T}}
    @latmap(eachindex(ψ, ϕ), scalar_mul_gpu!, ψ, ϕ, T(α))
    return nothing
end

@kernel cpu=false function scalar_mul_gpu!(ψ, ϕ, α, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    @inbounds ψ[site] = α * ϕ[site]
end

function LinearAlgebra.axpy!(α, ψ::TF, ϕ::TF) where {T,TF<:AnySpinorfield{<:GPU,T}}
    α = Complex{T}(α)
    @latmap(eachindex(ϕ, ψ), axpy_gpu!, ϕ, ψ, α)
    return nothing
end

@kernel cpu=false function axpy_gpu!(ϕ, @Const(ψ), α, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    @inbounds ϕ[site] += α * ψ[site]
end

function LinearAlgebra.axpby!(α, ψ::TF, β, ϕ::TF) where {T,TF<:AnySpinorfield{<:GPU,T}}
    α = Complex{T}(α)
    β = Complex{T}(β)
    @latmap(eachindex(ψ, ϕ), axpby_gpu!, ϕ, ψ, α, β)
    return nothing
end

@kernel cpu=false function axpby_gpu!(ϕ, @Const(ψ), α, β, bulk)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]
    @inbounds ϕ[site] = α * ψ[site] + β * ϕ[site]
end

function LinearAlgebra.dot(ϕ::TF, ψ::TF) where {TF<:AnySpinorfield{<:GPU}}
    return @latsum(eachindex(ϕ, ψ), ComplexF64, dot_gpu, ϕ, ψ)
end

@kernel cpu=false function dot_gpu(out, ϕ, ψ, bulk)
    iblock = @index(Group, Linear)
    iglobal = @index(Global, Cartesian)
    site = bulk[iglobal]

    resₙ = dot(ϕ[site], ψ[site])
    out_group = @groupreduce(+, resₙ, 0.0 + 0.0im)

    ithread = @index(Local)
    if ithread == 1
        @inbounds out[iblock] = out_group
    end
end

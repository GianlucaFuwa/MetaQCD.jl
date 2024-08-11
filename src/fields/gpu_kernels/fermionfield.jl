const AnyFermionfield{B,T,A,ND} = Union{Fermionfield{B,T,A,ND},EvenOdd{B,T,A,ND}}

function clear!(ϕ::AnyFermionfield{B}) where {B<:GPU}
    @latmap(Sequential(), Val(1), clear_fermion_kernel!, ϕ)
end

@kernel function clear_fermion_kernel!(ϕ)
    site = @index(Global, Cartesian)
    @inbounds ϕ[site] = zero(ϕ[site])
end

function Base.copy!(a::TF, b::TF) where {TF<:AnyFermionfield{<:GPU}}
    check_dims(a, b)
    @latmap(Sequential(), Val(1), copy_fermion_kernel!, a, b)
    return nothing
end

@kernel function copy_fermion_kernel!(a, @Const(b))
    site = @index(Global, Cartesian)
    @inbounds a[site] = b[site]
end

function ones!(ϕ::AnyFermionfield{B}) where {B<:GPU}
    @latmap(Sequential(), Val(1), ones_fermion_kernel!, ϕ)
    return nothing
end

@kernel function ones_fermion_kernel!(ϕ)
    site = @index(Global, Cartesian)
    @inbounds ϕ[site] = fill(1, ϕ[site])
end

function set_source!(ϕ::AnyFermionfield{B,T}, site, a, μ) where {B<:GPU,T}
    NC = num_colors(ϕ)
    ND = num_dirac(ϕ)
    @assert μ ∈ 1:ND && a ∈ 1:NC
    @latmap(Sequential(), Val(1), set_source_kernel!, ϕ, site, a, μ, NC, ND, T)
    return nothing
end

@kernel function set_source_kernel!(ϕ, site, a, μ, NC, ND, ::Type{T}) where {T}
    gsite = @index(Global, Cartesian)
    if gsite == site
        vec_index = (μ - 1) * NC + a
        tup = ntuple(i -> i == vec_index ? one(Complex{T}) : zero(Complex{T}), Val(3ND))
        @inbounds ϕ[site] = SVector{3ND,Complex{T}}(tup)
    else
        @inbounds ϕ[site] = zero(ϕ[site])
    end
end

function gaussian_pseudofermions!(ϕ::AnyFermionfield{B,T,A,ND}) where {B<:GPU,T,A,ND}
    @latmap(Sequential(), Val(1), gaussian_pseudofermions_kernel!, ϕ, Val(3ND), T)
    return nothing
end

@kernel function gaussian_pseudofermions_kernel!(ϕ, ::Val{L}, ::Type{T}) where {L,T}
    site = @index(Global, Cartesian)
    @inbounds ϕ[site] = @SVector randn(Complex{T}, L) # σ = 0.5
end

function LinearAlgebra.mul!(ϕ::AnyFermionfield{B,T,A,ND}, α) where {B<:GPU,T,A,ND}
    @latmap(Sequential(), Val(1), scalar_mul_kernel!, ϕ, T(α))
    return nothing
end

@kernel function scalar_mul_kernel!(ϕ, α)
    site = @index(Global, Cartesian)
    @inbounds ϕ[site] *= α
end

function LinearAlgebra.axpy!(α, ψ::TF, ϕ::TF) where {T,TF<:AnyFermionfield{<:GPU,T}}
    check_dims(ψ, ϕ)
    α = Complex{T}(α)
    @latmap(Sequential(), Val(1), axpy_kernel!, ϕ, ψ, α)
    return nothing
end

@kernel function axpy_kernel!(ϕ, @Const(ψ), α)
    site = @index(Global, Cartesian)
    @inbounds ϕ[site] += α * ψ[site]
end

function LinearAlgebra.axpby!(α, ψ::TF, β, ϕ::TF) where {T,TF<:AnyFermionfield{<:GPU,T}}
    check_dims(ψ, ϕ)
    α = Complex{T}(α)
    β = Complex{T}(β)
    @latmap(Sequential(), Val(1), axpby_kernel!, ϕ, ψ, α, β)
    return nothing
end

@kernel function axpby_kernel!(ϕ, @Const(ψ), α, β)
    site = @index(Global, Cartesian)
    @inbounds ϕ[site] = α * ψ[site] + β * ϕ[site]
end

function LinearAlgebra.dot(ϕ::TF, ψ::TF) where {TF<:AnyFermionfield{<:GPU}}
    check_dims(ψ, ϕ)
    return @latsum(Sequential(), Val(1), ComplexF64, dot_kernel, ϕ, ψ)
end

@kernel function dot_kernel(out, @Const(ϕ), @Const(ψ))
    bi = @index(Group, Linear)
    site = @index(Global, Cartesian)

    resₙ = 0.0 + 0.0im
    resₙ += cdot(ϕ[site], ψ[site])

    out_group = @groupreduce(+, resₙ, 0.0)

    ti = @index(Local)
    if ti == 1
        @inbounds out[bi] = out_group
    end
end

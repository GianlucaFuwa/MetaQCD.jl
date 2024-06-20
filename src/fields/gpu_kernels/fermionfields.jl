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

function gaussian_pseudofermions!(ϕ::AnyFermionfield{B,T}) where {B<:GPU,T}
    sz = num_dirac(ϕ) * num_colors(ϕ)
    @latmap(Sequential(), Val(1), gaussian_pseudofermions_kernel!, ϕ, sz)
    return nothing
end

@kernel function gaussian_pseudofermions_kernel!(ϕ, sz)
    site = @index(Global, Cartesian)
    @inbounds ϕ[site] = @SVector randn(Complex{T}, sz) # σ = 0.5
end

function LinearAlgebra.axpy!(α, ψ::TF, ϕ::TF) where {TF<:AnyFermionfield{<:GPU}}
    check_dims(ψ, ϕ)
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)
    @latmap(Sequential(), Val(1), axpy_kernel!, ϕ, ψ, α)
    return nothing
end

@kernel function axpy_kernel!(ϕ, @Const(ψ), α)
    site = @index(Global, Cartesian)
    @inbounds ϕ[site] += α * ψ[site]
end

function LinearAlgebra.axpby!(α, ψ::TF, β, ϕ::TF) where {TF<:AnyFermionfield{<:GPU}}
    check_dims(ψ, ϕ)
    FloatT = float_type(ϕ)
    α = Complex{FloatT}(α)
    β = Complex{FloatT}(β)
    @latmap(Sequential(), Val(1), axpby_kernel!, ϕ, ψ, α, β)
    return nothing
end

@kernel function axpy_kernel!(ϕ, @Const(ψ), α, β)
    site = @index(Global, Cartesian)
    @inbounds ϕ[site] = α * ψ[site] + β * ϕ[site]
end

function LinearAlgebra.dot(ϕ::TF, ψ::TF) where {TF<:AnyFermionfield{<:GPU}}
    check_dims(ψ, ϕ)
    return @latsum(Sequential(), Val(1), dot_kernel, ϕ, ψ)
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

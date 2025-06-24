function Base.copy!(a::TF, b::TF) where {TF<:SpinorfieldEO{<:GPU}}
    even_half = true
    @latmap(eachindex(even_half, a, b), copy_fermion_gpu!, a, b)
    return nothing
end

function gaussian_pseudofermions!(ϕ::SpinorfieldEO{B,T,M,A,ND}) where {B<:GPU,T,M,A,ND}
    even_half = true
    @latmap(eachindex(even_half, ϕ), gaussian_pseudofermions_gpu!, ϕ, Val(3ND), T)
    return nothing
end

function LinearAlgebra.mul!(ψ::TF, ϕ::TF, α) where {T,TF<:SpinorfieldEO{<:GPU,T}}
    even_half = true
    @latmap(eachindex(even_half, ψ, ϕ), scalar_mul_gpu!, ψ, ϕ, T(α))
    return nothing
end

function LinearAlgebra.axpy!(α, ψ::TF, ϕ::TF) where {T,TF<:SpinorfieldEO{<:GPU,T}}
    α = Complex{T}(α)
    even_half = true
    @latmap(eachindex(even_half, ϕ, ψ), axpy_gpu!, ϕ, ψ, α)
    return nothing
end

function LinearAlgebra.axpby!(α, ψ::TF, β, ϕ::TF) where {T,TF<:SpinorfieldEO{<:GPU,T}}
    α = Complex{T}(α)
    β = Complex{T}(β)
    even_half = true
    @latmap(eachindex(even_half, ψ, ϕ), axpby_gpu!, ϕ, ψ, α, β)
    return nothing
end

function LinearAlgebra.dot(ϕ::TF, ψ::TF) where {TF<:SpinorfieldEO{<:GPU}}
    even_half = true
    return @latsum(eachindex(even_half, ϕ, ψ), ComplexF64, dot_gpu, ϕ, ψ)
end

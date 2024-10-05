struct PauliMatrix{T<:AbstractFloat}
    upper::SMatrix{6,6,Complex{T},36}
    lower::SMatrix{6,6,Complex{T},36}
    function PauliMatrix(λ::UniformScaling{T}) where {T<:AbstractFloat}
        upper = lower = @SMatrix(zeros(Complex{T}, 6, 6)) + λ
        return new{T}(upper, lower)
    end

    function PauliMatrix(
        upper::SMatrix{6,6,Complex{T},36}, lower::SMatrix{6,6,Complex{T},36}
    ) where {T<:AbstractFloat}
        return new{T}(upper, lower)
    end
end

Base.zero(::Type{PauliMatrix{T}}) where {T} = PauliMatrix(UniformScaling(zero(T)))
Base.one(::Type{PauliMatrix{T}}) where {T} = PauliMatrix(UniformScaling(one(T)))

function rand_pauli(::Type{PauliMatrix{T}}) where {T}
    upper = hermitian(@SMatrix(rand(Complex{T}, 6, 6)))
    lower = hermitian(@SMatrix(rand(Complex{T}, 6, 6)))
    return PauliMatrix(upper, lower)
end

struct Pauli{T<:AbstractFloat}
    data::NTuple{36,T}
    Pauli(x::NTuple{36,T}) where {T<:AbstractFloat} = new{T}(x)

    function Pauli(λ::UniformScaling{T}) where {T<:AbstractFloat}
        c = λ.λ
        rest = ntuple(T(0.0), Val(30))
        x = (c, c, c, c, c, rest...)
        return new{T}(x)
    end
end

Base.zeros(::Type{Pauli{T}}) where {T} = Pauli(ntuple(_ -> T(0.0), Val(36)))

function pauli_mat(p::Pauli{T}) where {T}
    out = MMatrix{6,6,Complex{T},36}(undef)


end

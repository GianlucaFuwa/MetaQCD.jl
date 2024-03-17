"""
    cmvmul(A, x)

Return the matrix-vector product of `A` and `x`
"""
@inline function cmvmul(
    A::SMatrix{M,N,Complex{T},NM}, x::SVector{N,Complex{T}}
) where {T,N,M,NM}
    return SVector(cmvmul!(MVector{M,Complex{T}}(undef), MMatrix(A), MVector(x)))
end

@inline function cmvmul!(
    yc::MVector{M,Complex{T}}, Ac::MMatrix{M,N,Complex{T},NM}, xc::MVector{N,Complex{T}}
) where {T,N,M,NM}
    y = reinterpret(reshape, T, yc)
    A = reinterpret(reshape, T, Ac)
    x = reinterpret(reshape, T, xc)

    @turbo for m in Base.Slice(static(1):static(M))
        yre = zero(T)
        yim = zero(T)
        for n in Base.Slice(static(1):static(N))
            yre += A[1, m, n] * x[1, n] - A[2, m, n] * x[2, n]
            yim += A[1, m, n] * x[2, n] + A[2, m, n] * x[1, n]
        end
        y[1, m] = yre
        y[2, m] = yim
    end

    return yc
end

"""
    cmvmul_d(A, x)

Return the matrix-vector product of the adjoint of `A` and `x`
"""
@inline function cmvmul_d(
    A::SMatrix{M,N,Complex{T},NM}, x::SVector{N,Complex{T}}
) where {T,N,M,NM}
    return SVector(cmvmul_d!(MVector{M,Complex{T}}(undef), MMatrix(A), MVector(x)))
end

@inline function cmvmul_d!(
    yc::MVector{M,Complex{T}}, Ac::MMatrix{M,N,Complex{T},NM}, xc::MVector{N,Complex{T}}
) where {T,N,M,NM}
    y = reinterpret(reshape, T, yc)
    A = reinterpret(reshape, T, Ac)
    x = reinterpret(reshape, T, xc)

    @turbo for m in Base.Slice(static(1):static(M))
        yre = zero(T)
        yim = zero(T)
        for n in Base.Slice(static(1):static(N))
            yre += A[1, n, m] * x[1, n] + A[2, n, m] * x[2, n]
            yim += A[1, n, m] * x[2, n] - A[2, n, m] * x[1, n]
        end
        y[1, m] = yre
        y[2, m] = yim
    end

    return yc
end

"""
    cvmmul(x, A)

Return the vector-matrix product of `x` and `A`. `x` is implicitly assumed to be a column
vector and therefore the adjoint of `x` is used
"""
@inline function cvmmul(
    x::SVector{N,Complex{T}}, A::SMatrix{M,N,Complex{T},NM}
) where {T,N,M,NM}
    return SVector(cvmmul!(MVector{M,Complex{T}}(undef), MVector(x), MMatrix(A)))
end

@inline function cvmmul!(
    yc::MVector{M,Complex{T}}, xc::MVector{N,Complex{T}}, Ac::MMatrix{M,N,Complex{T},NM}
) where {T,N,M,NM}
    y = reinterpret(reshape, T, yc)
    A = reinterpret(reshape, T, Ac)
    x = reinterpret(reshape, T, xc)

    @turbo for m in Base.Slice(static(1):static(M))
        yre = zero(T)
        yim = zero(T)
        for n in Base.Slice(static(1):static(N))
            yre += A[1, m, n] * x[1, n] + A[2, m, n] * x[2, n]
            yim += A[1, m, n] * x[2, n] - A[2, m, n] * x[1, n]
        end
        y[1, m] = yre
        y[2, m] = yim
    end

    return yc
end

"""
    cvmmul_d(x, A)

Return the vector-matrix product of `x` and the adjoint of `A`. `x` is implicitly assumed
to be a column vector and therefore the adjoint of `x` is used
"""
@inline function cvmmul_d(
    x::SVector{N,Complex{T}}, A::SMatrix{M,N,Complex{T},NM}
) where {T,N,M,NM}
    return SVector(cmvmul_d!(MVector{M,Complex{T}}(undef), MVector(x), MMatrix(A)))
end

@inline function cvmmul_d!(
    yc::MVector{M,Complex{T}}, xc::MVector{N,Complex{T}}, Ac::MMatrix{M,N,Complex{T},NM}
) where {T,N,M,NM}
    y = reinterpret(reshape, T, yc)
    A = reinterpret(reshape, T, Ac)
    x = reinterpret(reshape, T, xc)

    @turbo for m in Base.Slice(static(1):static(M))
        yre = zero(T)
        yim = zero(T)
        for n in Base.Slice(static(1):static(N))
            yre += A[1, n, m] * x[1, n] - A[2, n, m] * x[2, n]
            yim += A[1, n, m] * x[2, n] + A[2, n, m] * x[1, n]
        end
        y[1, m] = yre
        y[2, m] = yim
    end

    return yc
end

"""
    cdot(a, b)

Return the complex dot product of `a` and `b`
"""
@inline function cdot(a::SVector{N,Complex{T}}, b::SVector{N,Complex{T}}) where {T,N}
    return cdot(MVector(a), MVector(b))
end

@inline function cdot(ac::MVector{N,Complex{T}}, bc::MVector{N,Complex{T}}) where {T,N}
    a = reinterpret(reshape, T, ac)
    b = reinterpret(reshape, T, bc)
    r = zero(T)
    i = zero(T)

    @turbo for n in Base.Slice(static(1):static(N))
        r += a[1, n] * b[1, n] + a[2, n] * b[2, n]
        i += a[1, n] * b[2, n] - a[2, n] * b[1, n]
    end

    return Complex{T}(r, i)
end

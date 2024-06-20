#================================================================================#
# This file implements the low-level linear algebra operations for complex
# matrix-vector and vector-vector operations, most notably the m-v/v-m product, 
# the complex dot product, and the complex Kronecker product.
#
# All implementations make use of the @turbo loop-vectorization macro, which 
# unrolls the loops and applies AVX/AVX2/AVX512 SIMD instructions while employing a 
# certain degree of cost modelling and thereby achieves better peroformance than
# the LLVM auto-vectorizer, especially when vector lengths are not a multiple of
# 256 bits.
#================================================================================#

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

Return the vector-matrix product of `x†` and `A`.
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
            yre += A[1, n, m] * x[1, n] + A[2, n, m] * x[2, n]
            yim -= A[1, n, m] * x[2, n] - A[2, n, m] * x[1, n]
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
    return SVector(cvmmul_d!(MVector{M,Complex{T}}(undef), MVector(x), MMatrix(A)))
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
            yre += A[1, m, n] * x[1, n] - A[2, m, n] * x[2, n]
            yim -= A[1, m, n] * x[2, n] + A[2, m, n] * x[1, n]
        end
        y[1, m] = yre
        y[2, m] = yim
    end

    return yc
end

"""
    cmvmul_color(A, x)

Return the matrix-vector product of `A` and `x`, where `A` only acts on the color structure
of `x`.
"""
@inline function cmvmul_color(
    A::SMatrix{N,N,Complex{T},N2}, x::SVector{M,Complex{T}}
) where {T,N,M,N2}
    return SVector(cmvmul_color!(MVector{M,Complex{T}}(undef), MMatrix(A), MVector(x)))
end

@generated function cmvmul_color!(
    yc::MVector{M,Complex{T}}, Ac::MMatrix{N,N,Complex{T},N2}, xc::MVector{M,Complex{T}}
) where {T,N,M,N2}
    if M ÷ N !== 4
        return :(throw(DimensionMismatch("length(x) must be 4 times the side length of A")))
    end

    q = quote
        $(Expr(:meta, :inline))
        y = reinterpret(reshape, $T, yc)
        A = reinterpret(reshape, $T, Ac)
        x = reinterpret(reshape, $T, xc)
    end

    loop_q = quote
        @turbo for m in Base.Slice(static(1):static($N))
            yₘ₁r = $(zero(T))
            yₘ₁i = $(zero(T))
            yₘ₂r = $(zero(T))
            yₘ₂i = $(zero(T))
            yₘ₃r = $(zero(T))
            yₘ₃i = $(zero(T))
            yₘ₄r = $(zero(T))
            yₘ₄i = $(zero(T))
            for n in Base.Slice(static(1):static($N))
                Aₘₙr = A[1, m, n]
                Aₘₙi = A[2, m, n]
                xₙ₁r = x[1, n]
                xₙ₁i = x[2, n]
                xₙ₂r = x[1, $N+n]
                xₙ₂i = x[2, $N+n]
                xₙ₃r = x[1, $(2N)+n]
                xₙ₃i = x[2, $(2N)+n]
                xₙ₄r = x[1, $(3N)+n]
                xₙ₄i = x[2, $(3N)+n]
                yₘ₁r += Aₘₙr * xₙ₁r - Aₘₙi * xₙ₁i
                yₘ₁i += Aₘₙr * xₙ₁i + Aₘₙi * xₙ₁r
                yₘ₂r += Aₘₙr * xₙ₂r - Aₘₙi * xₙ₂i
                yₘ₂i += Aₘₙr * xₙ₂i + Aₘₙi * xₙ₂r
                yₘ₃r += Aₘₙr * xₙ₃r - Aₘₙi * xₙ₃i
                yₘ₃i += Aₘₙr * xₙ₃i + Aₘₙi * xₙ₃r
                yₘ₄r += Aₘₙr * xₙ₄r - Aₘₙi * xₙ₄i
                yₘ₄i += Aₘₙr * xₙ₄i + Aₘₙi * xₙ₄r
            end
            y[1, m] = yₘ₁r
            y[2, m] = yₘ₁i
            y[1, $N+m] = yₘ₂r
            y[2, $N+m] = yₘ₂i
            y[1, $(2N)+m] = yₘ₃r
            y[2, $(2N)+m] = yₘ₃i
            y[1, $(3N)+m] = yₘ₄r
            y[2, $(3N)+m] = yₘ₄i
        end
        return yc
    end

    push!(q.args, loop_q)
    return q
end

"""
    cmvmul_d_color(A, x)

Return the matrix-vector product of `A†` and `x`, where `A†` only acts on the colo
structure of `x`.
"""
@inline function cmvmul_d_color(
    A::SMatrix{N,N,Complex{T},N2}, x::SVector{M,Complex{T}}
) where {T,N,M,N2}
    return SVector(cmvmul_d_color!(MVector{M,Complex{T}}(undef), MMatrix(A), MVector(x)))
end

@generated function cmvmul_d_color!(
    yc::MVector{M,Complex{T}}, Ac::MMatrix{N,N,Complex{T},N2}, xc::MVector{M,Complex{T}}
) where {T,N,M,N2}
    if M ÷ N !== 4
        return :(throw(DimensionMismatch("length(x) must be 4 times the side length of A")))
    end

    q = quote
        $(Expr(:meta, :inline))
        y = reinterpret(reshape, $T, yc)
        A = reinterpret(reshape, $T, Ac)
        x = reinterpret(reshape, $T, xc)
    end

    loop_q = quote
        @turbo for m in Base.Slice(static(1):static($N))
            yₘ₁r = $(zero(T))
            yₘ₁i = $(zero(T))
            yₘ₂r = $(zero(T))
            yₘ₂i = $(zero(T))
            yₘ₃r = $(zero(T))
            yₘ₃i = $(zero(T))
            yₘ₄r = $(zero(T))
            yₘ₄i = $(zero(T))
            for n in Base.Slice(static(1):static($N))
                Aₙₘr = A[1, n, m]
                Aₙₘi = A[2, n, m]
                xₙ₁r = x[1, n]
                xₙ₁i = x[2, n]
                xₙ₂r = x[1, $N+n]
                xₙ₂i = x[2, $N+n]
                xₙ₃r = x[1, $(2N)+n]
                xₙ₃i = x[2, $(2N)+n]
                xₙ₄r = x[1, $(3N)+n]
                xₙ₄i = x[2, $(3N)+n]
                yₘ₁r += Aₙₘr * xₙ₁r + Aₙₘi * xₙ₁i
                yₘ₁i += Aₙₘr * xₙ₁i - Aₙₘi * xₙ₁r
                yₘ₂r += Aₙₘr * xₙ₂r + Aₙₘi * xₙ₂i
                yₘ₂i += Aₙₘr * xₙ₂i - Aₙₘi * xₙ₂r
                yₘ₃r += Aₙₘr * xₙ₃r + Aₙₘi * xₙ₃i
                yₘ₃i += Aₙₘr * xₙ₃i - Aₙₘi * xₙ₃r
                yₘ₄r += Aₙₘr * xₙ₄r + Aₙₘi * xₙ₄i
                yₘ₄i += Aₙₘr * xₙ₄i - Aₙₘi * xₙ₄r
            end
            y[1, m] = yₘ₁r
            y[2, m] = yₘ₁i
            y[1, $N+m] = yₘ₂r
            y[2, $N+m] = yₘ₂i
            y[1, $(2N)+m] = yₘ₃r
            y[2, $(2N)+m] = yₘ₃i
            y[1, $(3N)+m] = yₘ₄r
            y[2, $(3N)+m] = yₘ₄i
        end
        return yc
    end

    push!(q.args, loop_q)
    return q
end

"""
    cvmmul_color(x, A)

Return the matrix-vector product of `x†` and `A`, where `A` only acts on the color
structure of `x†`.
"""
@inline function cvmmul_color(
    x::SVector{M,Complex{T}}, A::SMatrix{N,N,Complex{T},N2}
) where {T,N,M,N2}
    return SVector(cvmmul_color!(MVector{M,Complex{T}}(undef), MVector(x), MMatrix(A)))
end

@generated function cvmmul_color!(
    yc::MVector{M,Complex{T}}, xc::MVector{M,Complex{T}}, Ac::MMatrix{N,N,Complex{T},N2}
) where {T,N,M,N2}
    if M ÷ N !== 4
        return :(throw(DimensionMismatch("length(x) must be 4 times the side length of A")))
    end

    q = quote
        $(Expr(:meta, :inline))
        y = reinterpret(reshape, $T, yc)
        A = reinterpret(reshape, $T, Ac)
        x = reinterpret(reshape, $T, xc)
    end

    loop_q = quote
        @turbo for m in Base.Slice(static(1):static($N))
            yₘ₁r = $(zero(T))
            yₘ₁i = $(zero(T))
            yₘ₂r = $(zero(T))
            yₘ₂i = $(zero(T))
            yₘ₃r = $(zero(T))
            yₘ₃i = $(zero(T))
            yₘ₄r = $(zero(T))
            yₘ₄i = $(zero(T))
            for n in Base.Slice(static(1):static($N))
                Aₙₘr = A[1, n, m]
                Aₙₘi = A[2, n, m]
                xₙ₁r = x[1, n]
                xₙ₁i = x[2, n]
                xₙ₂r = x[1, $N+n]
                xₙ₂i = x[2, $N+n]
                xₙ₃r = x[1, $(2N)+n]
                xₙ₃i = x[2, $(2N)+n]
                xₙ₄r = x[1, $(3N)+n]
                xₙ₄i = x[2, $(3N)+n]
                yₘ₁r += Aₙₘr * xₙ₁r + Aₙₘi * xₙ₁i
                yₘ₁i -= Aₙₘr * xₙ₁i - Aₙₘi * xₙ₁r
                yₘ₂r += Aₙₘr * xₙ₂r + Aₙₘi * xₙ₂i
                yₘ₂i -= Aₙₘr * xₙ₂i - Aₙₘi * xₙ₂r
                yₘ₃r += Aₙₘr * xₙ₃r + Aₙₘi * xₙ₃i
                yₘ₃i -= Aₙₘr * xₙ₃i - Aₙₘi * xₙ₃r
                yₘ₄r += Aₙₘr * xₙ₄r + Aₙₘi * xₙ₄i
                yₘ₄i -= Aₙₘr * xₙ₄i - Aₙₘi * xₙ₄r
            end
            y[1, m] = yₘ₁r
            y[2, m] = yₘ₁i
            y[1, $N+m] = yₘ₂r
            y[2, $N+m] = yₘ₂i
            y[1, $(2N)+m] = yₘ₃r
            y[2, $(2N)+m] = yₘ₃i
            y[1, $(3N)+m] = yₘ₄r
            y[2, $(3N)+m] = yₘ₄i
        end
        return yc
    end

    push!(q.args, loop_q)
    return q
end

"""
    cvmmul_d_color(x, A)

Return the matrix-vector product of `x†` and `A†`, where `A†` only acts on the color
structure of `x†`.
"""
@inline function cvmmul_d_color(
    x::SVector{M,Complex{T}}, A::SMatrix{N,N,Complex{T},N2}
) where {T,N,M,N2}
    return SVector(cvmmul_d_color!(MVector{M,Complex{T}}(undef), MVector(x), MMatrix(A)))
end

@generated function cvmmul_d_color!(
    yc::MVector{M,Complex{T}}, xc::MVector{M,Complex{T}}, Ac::MMatrix{N,N,Complex{T},N2}
) where {T,N,M,N2}
    if M ÷ N !== 4
        return :(throw(DimensionMismatch("length(x) must be 4 times the side length of A")))
    end

    q = quote
        $(Expr(:meta, :inline))
        y = reinterpret(reshape, $T, yc)
        A = reinterpret(reshape, $T, Ac)
        x = reinterpret(reshape, $T, xc)
    end

    loop_q = quote
        @turbo for m in Base.Slice(static(1):static($N))
            yₘ₁r = $(zero(T))
            yₘ₁i = $(zero(T))
            yₘ₂r = $(zero(T))
            yₘ₂i = $(zero(T))
            yₘ₃r = $(zero(T))
            yₘ₃i = $(zero(T))
            yₘ₄r = $(zero(T))
            yₘ₄i = $(zero(T))
            for n in Base.Slice(static(1):static($N))
                Aₙₘr = A[1, m, n]
                Aₙₘi = A[2, m, n]
                xₙ₁r = x[1, n]
                xₙ₁i = x[2, n]
                xₙ₂r = x[1, $N+n]
                xₙ₂i = x[2, $N+n]
                xₙ₃r = x[1, $(2N)+n]
                xₙ₃i = x[2, $(2N)+n]
                xₙ₄r = x[1, $(3N)+n]
                xₙ₄i = x[2, $(3N)+n]
                yₘ₁r += Aₙₘr * xₙ₁r - Aₙₘi * xₙ₁i
                yₘ₁i -= Aₙₘr * xₙ₁i + Aₙₘi * xₙ₁r
                yₘ₂r += Aₙₘr * xₙ₂r - Aₙₘi * xₙ₂i
                yₘ₂i -= Aₙₘr * xₙ₂i + Aₙₘi * xₙ₂r
                yₘ₃r += Aₙₘr * xₙ₃r - Aₙₘi * xₙ₃i
                yₘ₃i -= Aₙₘr * xₙ₃i + Aₙₘi * xₙ₃r
                yₘ₄r += Aₙₘr * xₙ₄r - Aₙₘi * xₙ₄i
                yₘ₄i -= Aₙₘr * xₙ₄i + Aₙₘi * xₙ₄r
            end
            y[1, m] = yₘ₁r
            y[2, m] = yₘ₁i
            y[1, $N+m] = yₘ₂r
            y[2, $N+m] = yₘ₂i
            y[1, $(2N)+m] = yₘ₃r
            y[2, $(2N)+m] = yₘ₃i
            y[1, $(3N)+m] = yₘ₄r
            y[2, $(3N)+m] = yₘ₄i
        end
        return yc
    end

    push!(q.args, loop_q)
    return q
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

"""
    ckron(a, b)
    ckron(A, B)

Return the complex Kronecker(outer) product of vectors `a` and `b`, i.e. `a ⊗ b†`, or of
two matrices `A` and `B`, i.e. `A ⊗ B`.
"""
@inline function ckron(a::SVector{M,Complex{T}}, b::SVector{N,Complex{T}}) where {T,M,N}
    return SMatrix(ckron!(MMatrix{M,N,Complex{T},M * N}(undef), MVector(a), MVector(b)))
end

@inline function ckron!(
    cc::MMatrix{M,N,Complex{T},MN}, ac::MVector{M,Complex{T}}, bc::MVector{N,Complex{T}}
) where {T,M,N,MN}
    c = reinterpret(reshape, T, cc)
    a = reinterpret(reshape, T, ac)
    b = reinterpret(reshape, T, bc)

    @turbo for m in Base.Slice(static(1):static(M))
        for n in Base.Slice(static(1):static(N))
            c[1, n, m] = a[1, n] * b[1, m] + a[2, n] * b[2, m]
            c[2, n, m] = -a[1, n] * b[2, m] + a[2, n] * b[1, m]
        end
    end

    return cc
end

@inline function ckron(
    A::SMatrix{N,M,Complex{T},NM}, B::SMatrix{K,L,Complex{T},KL}
) where {T,N,M,NM,K,L,KL}
    return SMatrix(
        ckron!(MMatrix{N * K,M * L,Complex{T},NM * KL}(undef), MMatrix(A), MMatrix(B))
    )
end

@inline function ckron!(
    Cc::MMatrix{NK,ML,Complex{T},NMKL},
    Ac::MMatrix{N,M,Complex{T},NM},
    Bc::MMatrix{K,L,Complex{T},KL},
) where {T,NK,ML,NMKL,N,M,NM,K,L,KL}
    C = reinterpret(reshape, T, Cc)
    A = reinterpret(reshape, T, Ac)
    B = reinterpret(reshape, T, Bc)

    @turbo for m in Base.Slice(static(1):static(M))
        for n in Base.Slice(static(1):static(N))
            for l in Base.Slice(static(1):static(L))
                for k in Base.Slice(static(1):static(K))
                    C[1, K*(n-1)+k, L*(m-1)+l] =
                        A[1, n, m] * B[1, k, l] - A[2, n, m] * B[2, k, l]
                    C[2, K*(n-1)+k, L*(m-1)+l] =
                        A[1, n, m] * B[2, k, l] + A[2, n, m] * B[1, k, l]
                end
            end
        end
    end

    return Cc
end

"""
    spintrace(a, b)

Return the complex Kronecker(outer) product of vectors `a` and `b`, summing over dirac
indices, i.e. `∑ ᵨ aᵨ ⊗ bᵨ†`
"""
@inline function spintrace(a::SVector{M,Complex{T}}, b::SVector{M,Complex{T}}) where {T,M}
    @assert M % 4 == 0 "length of inputs must be a multiple of 4"
    N = M ÷ 4
    return SMatrix(spintrace!(MMatrix{N,N,Complex{T},N * N}(undef), MVector(a), MVector(b)))
end

@inline function spintrace!(
    cc::MMatrix{N,N,Complex{T},NN}, ac::MVector{M,Complex{T}}, bc::MVector{M,Complex{T}}
) where {T,M,N,NN}
    c = reinterpret(reshape, T, cc)
    a = reinterpret(reshape, T, ac)
    b = reinterpret(reshape, T, bc)

    for m in Base.Slice(static(1):static(N))
        for n in Base.Slice(static(1):static(N))
            cre = zero(T)
            cim = zero(T)
            for k in Base.Slice(static(1):static(4))
                cre += a[1, N*(k-1)+n] * b[1, N*(k-1)+m] + a[2, N*(k-1)+n] * b[2, N*(k-1)+m]
                cim -= a[1, N*(k-1)+n] * b[2, N*(k-1)+m] - a[2, N*(k-1)+n] * b[1, N*(k-1)+m]
            end
            c[1, n, m] = cre
            c[2, n, m] = cim
        end
    end

    return cc
end

"""
    spin_proj(x, ::Val{ρ})

Return `(1 ± γᵨ) * x` where `γᵨ` is the ρ-th Euclidean gamma matrix in the Chiral basis.
and `x` is a 4xN component complex vector. The second argument is ρ wrapped in a `Val` and
must be within the range [-4,4]. Its sign determines the sign in front of the γᵨ matrix.
"""
@inline function spin_proj(x::SVector{M,Complex{T}}, ::Val{ρ}) where {T,M,ρ}
    return SVector(spin_proj!(MVector{M,Complex{T}}(undef), MVector(x), Val(ρ)))
end

@generated function spin_proj!(
    yc::MVector{M,Complex{T}}, xc::MVector{M,Complex{T}}, ::Val{ρ}
) where {T,M,ρ}
    if M % 4 != 0
        return :(throw(DimensionMismatch("length(x) must be a multiple of 4")))
    end

    N = M ÷ 4

    q = quote
        $(Expr(:meta, :inline))
        y = reinterpret(reshape, $T, yc)
        x = reinterpret(reshape, $T, xc)
    end

    if ρ === 1
        calc_hi = quote
            xₙ₁r = x[1, m] + x[2, $(3N)+m]
            xₙ₁i = x[2, m] - x[1, $(3N)+m]
            xₙ₂r = x[1, $N+m] + x[2, $(2N)+m]
            xₙ₂i = x[2, $N+m] - x[1, $(2N)+m]
        end
        set_lo = quote
            y[1, $(2N)+m] = -xₙ₂i
            y[2, $(2N)+m] = xₙ₂r
            y[1, $(3N)+m] = -xₙ₁i
            y[2, $(3N)+m] = xₙ₁r
        end
    elseif ρ === -1
        calc_hi = quote
            xₙ₁r = x[1, m] - x[2, $(3N)+m]
            xₙ₁i = x[2, m] + x[1, $(3N)+m]
            xₙ₂r = x[1, $N+m] - x[2, $(2N)+m]
            xₙ₂i = x[2, $N+m] + x[1, $(2N)+m]
        end
        set_lo = quote
            y[1, $(2N)+m] = xₙ₂i
            y[2, $(2N)+m] = -xₙ₂r
            y[1, $(3N)+m] = xₙ₁i
            y[2, $(3N)+m] = -xₙ₁r
        end
    elseif ρ === 2
        calc_hi = quote
            xₙ₁r = x[1, m] - x[1, $(3N)+m]
            xₙ₁i = x[2, m] - x[2, $(3N)+m]
            xₙ₂r = x[1, $N+m] + x[1, $(2N)+m]
            xₙ₂i = x[2, $N+m] + x[2, $(2N)+m]
        end
        set_lo = quote
            y[1, $(2N)+m] = xₙ₂r
            y[2, $(2N)+m] = xₙ₂i
            y[1, $(3N)+m] = -xₙ₁r
            y[2, $(3N)+m] = -xₙ₁i
        end
    elseif ρ === -2
        calc_hi = quote
            xₙ₁r = x[1, m] + x[1, $(3N)+m]
            xₙ₁i = x[2, m] + x[2, $(3N)+m]
            xₙ₂r = x[1, $N+m] - x[1, $(2N)+m]
            xₙ₂i = x[2, $N+m] - x[2, $(2N)+m]
        end
        set_lo = quote
            y[1, $(2N)+m] = -xₙ₂r
            y[2, $(2N)+m] = -xₙ₂i
            y[1, $(3N)+m] = xₙ₁r
            y[2, $(3N)+m] = xₙ₁i
        end
    elseif ρ === 3
        calc_hi = quote
            xₙ₁r = x[1, m] + x[2, $(2N)+m]
            xₙ₁i = x[2, m] - x[1, $(2N)+m]
            xₙ₂r = x[1, $N+m] - x[2, $(3N)+m]
            xₙ₂i = x[2, $N+m] + x[1, $(3N)+m]
        end
        set_lo = quote
            y[1, $(2N)+m] = -xₙ₁i
            y[2, $(2N)+m] = xₙ₁r
            y[1, $(3N)+m] = xₙ₂i
            y[2, $(3N)+m] = -xₙ₂r
        end
    elseif ρ === -3
        calc_hi = quote
            xₙ₁r = x[1, m] - x[2, $(2N)+m]
            xₙ₁i = x[2, m] + x[1, $(2N)+m]
            xₙ₂r = x[1, $N+m] + x[2, $(3N)+m]
            xₙ₂i = x[2, $N+m] - x[1, $(3N)+m]
        end
        set_lo = quote
            y[1, $(2N)+m] = xₙ₁i
            y[2, $(2N)+m] = -xₙ₁r
            y[1, $(3N)+m] = -xₙ₂i
            y[2, $(3N)+m] = xₙ₂r
        end
    elseif ρ === 4
        calc_hi = quote
            xₙ₁r = x[1, m] + x[1, $(2N)+m]
            xₙ₁i = x[2, m] + x[2, $(2N)+m]
            xₙ₂r = x[1, $N+m] + x[1, $(3N)+m]
            xₙ₂i = x[2, $N+m] + x[2, $(3N)+m]
        end
        set_lo = quote
            y[1, $(2N)+m] = xₙ₁r
            y[2, $(2N)+m] = xₙ₁i
            y[1, $(3N)+m] = xₙ₂r
            y[2, $(3N)+m] = xₙ₂i
        end
    elseif ρ === -4
        calc_hi = quote
            xₙ₁r = x[1, m] - x[1, $(2N)+m]
            xₙ₁i = x[2, m] - x[2, $(2N)+m]
            xₙ₂r = x[1, $N+m] - x[1, $(3N)+m]
            xₙ₂i = x[2, $N+m] - x[2, $(3N)+m]
        end
        set_lo = quote
            y[1, $(2N)+m] = -xₙ₁r
            y[2, $(2N)+m] = -xₙ₁i
            y[1, $(3N)+m] = -xₙ₂r
            y[2, $(3N)+m] = -xₙ₂i
        end
    else
        return :(throw(DimensionMismatch("ρ must be in [-4,4]")))
    end

    loop_q = quote
        @turbo for m in Base.Slice(static(1):static($N))
            $calc_hi
            y[1, m] = xₙ₁r
            y[2, m] = xₙ₁i
            y[1, $N+m] = xₙ₂r
            y[2, $N+m] = xₙ₂i
            $set_lo
        end
        return yc
    end

    push!(q.args, loop_q)
    return q
end

"""
    cmvmul_spin_proj(A, x, ::Val{ρ}, ::Val{is_adjoint}=Val(false))

Return `A * (1 ± γᵨ) * x` where `γᵨ` is the ρ-th Euclidean gamma matrix in the Chiral
basis. `x` is assumed to be a 4xN component complex vector. The third argument is ρ wrapped
in a `Val` and must be within the range [-4,4]. Its sign determines the sign in front of the
γᵨ matrix. If `is_adjoint` is `true`, `A†` is used instead of `A`.
"""
@inline function cmvmul_spin_proj(
    A::SMatrix{N,N,Complex{T},N2},
    x::SVector{M,Complex{T}},
    ::Val{ρ},
    ::Val{is_adjoint}=Val(false),
) where {T,N,M,N2,ρ,is_adjoint}
    return SVector(
        cmvmul_spin_proj!(
            MVector{M,Complex{T}}(undef),
            MMatrix(A),
            MVector(x),
            MVector{2N,Complex{T}}(undef),
            Val(ρ),
            Val(is_adjoint),
        ),
    )
end

# We use a generated function here, which means that the function is generated at compile
# time / first call time. We do this here to avoid if-checks on ρ and is_adjoint, since we
# can pass them as type parameters using the `Val` functionality, which wraps a value as a
# type.
@generated function cmvmul_spin_proj!(
    yc::MVector{M,Complex{T}},
    Ac::MMatrix{N,N,Complex{T},N2},
    xc::MVector{M,Complex{T}},
    uc::MVector{L,Complex{T}},
    ::Val{ρ},
    ::Val{is_adjoint},
) where {T,N,M,N2,L,ρ,is_adjoint}
    # An important thing to note about generated functions is that no information about the
    # actual values of the inputs are known, only their types. This is fine, since we pass
    # the dimensions of the input matrices and vectors as type parameters and so we can use
    # them here
    # Generated funtions return an expression, which is then evaluated at run time
    if M ÷ N !== 4
        return :(throw(DimensionMismatch("length(x) must be 4 times the side length of A")))
    end

    q = quote
        $(Expr(:meta, :inline))
        y = reinterpret(reshape, $T, yc)
        A = reinterpret(reshape, $T, Ac)
        x = reinterpret(reshape, $T, xc)
        u = reinterpret(reshape, $T, uc)
    end

    # We pack calculations into expressions, which are basically just code snippets that
    # can be executed. We can then push these expressions into any other expression
    # (like a quote block) and they will be executed in order. We can interpolate variables
    # into these expressions using the $-syntax. Any operations within an interpolation
    # are also executed at compile time, e.g., the $(3N) == $(3 * N) below
    # Note that all the if-else branches are generated at compile time, so only the
    # relevant branch is compiled into the final function
    if ρ === 1
        calc_hi = quote
            xₙ₁r = x[1, n] + x[2, $(3N)+n]
            xₙ₁i = x[2, n] - x[1, $(3N)+n]
            xₙ₂r = x[1, $N+n] + x[2, $(2N)+n]
            xₙ₂i = x[2, $N+n] - x[1, $(2N)+n]
        end
        set_lo = quote
            y[1, $(2N)+m] = -yₘ₂i
            y[2, $(2N)+m] = yₘ₂r
            y[1, $(3N)+m] = -yₘ₁i
            y[2, $(3N)+m] = yₘ₁r
        end
    elseif ρ === -1
        calc_hi = quote
            xₙ₁r = x[1, n] - x[2, $(3N)+n]
            xₙ₁i = x[2, n] + x[1, $(3N)+n]
            xₙ₂r = x[1, $N+n] - x[2, $(2N)+n]
            xₙ₂i = x[2, $N+n] + x[1, $(2N)+n]
        end
        set_lo = quote
            y[1, $(2N)+m] = yₘ₂i
            y[2, $(2N)+m] = -yₘ₂r
            y[1, $(3N)+m] = yₘ₁i
            y[2, $(3N)+m] = -yₘ₁r
        end
    elseif ρ === 2
        calc_hi = quote
            xₙ₁r = x[1, n] - x[1, $(3N)+n]
            xₙ₁i = x[2, n] - x[2, $(3N)+n]
            xₙ₂r = x[1, $N+n] + x[1, $(2N)+n]
            xₙ₂i = x[2, $N+n] + x[2, $(2N)+n]
        end
        set_lo = quote
            y[1, $(2N)+m] = yₘ₂r
            y[2, $(2N)+m] = yₘ₂i
            y[1, $(3N)+m] = -yₘ₁r
            y[2, $(3N)+m] = -yₘ₁i
        end
    elseif ρ === -2
        calc_hi = quote
            xₙ₁r = x[1, n] + x[1, $(3N)+n]
            xₙ₁i = x[2, n] + x[2, $(3N)+n]
            xₙ₂r = x[1, $N+n] - x[1, $(2N)+n]
            xₙ₂i = x[2, $N+n] - x[2, $(2N)+n]
        end
        set_lo = quote
            y[1, $(2N)+m] = -yₘ₂r
            y[2, $(2N)+m] = -yₘ₂i
            y[1, $(3N)+m] = yₘ₁r
            y[2, $(3N)+m] = yₘ₁i
        end
    elseif ρ === 3
        calc_hi = quote
            xₙ₁r = x[1, n] + x[2, $(2N)+n]
            xₙ₁i = x[2, n] - x[1, $(2N)+n]
            xₙ₂r = x[1, $N+n] - x[2, $(3N)+n]
            xₙ₂i = x[2, $N+n] + x[1, $(3N)+n]
        end
        set_lo = quote
            y[1, $(2N)+m] = -yₘ₁i
            y[2, $(2N)+m] = yₘ₁r
            y[1, $(3N)+m] = yₘ₂i
            y[2, $(3N)+m] = -yₘ₂r
        end
    elseif ρ === -3
        calc_hi = quote
            xₙ₁r = x[1, n] - x[2, $(2N)+n]
            xₙ₁i = x[2, n] + x[1, $(2N)+n]
            xₙ₂r = x[1, $N+n] + x[2, $(3N)+n]
            xₙ₂i = x[2, $N+n] - x[1, $(3N)+n]
        end
        set_lo = quote
            y[1, $(2N)+m] = yₘ₁i
            y[2, $(2N)+m] = -yₘ₁r
            y[1, $(3N)+m] = -yₘ₂i
            y[2, $(3N)+m] = yₘ₂r
        end
    elseif ρ === 4
        calc_hi = quote
            xₙ₁r = x[1, n] + x[1, $(2N)+n]
            xₙ₁i = x[2, n] + x[2, $(2N)+n]
            xₙ₂r = x[1, $N+n] + x[1, $(3N)+n]
            xₙ₂i = x[2, $N+n] + x[2, $(3N)+n]
        end
        set_lo = quote
            y[1, $(2N)+m] = yₘ₁r
            y[2, $(2N)+m] = yₘ₁i
            y[1, $(3N)+m] = yₘ₂r
            y[2, $(3N)+m] = yₘ₂i
        end
    elseif ρ === -4
        calc_hi = quote
            xₙ₁r = x[1, n] - x[1, $(2N)+n]
            xₙ₁i = x[2, n] - x[2, $(2N)+n]
            xₙ₂r = x[1, $N+n] - x[1, $(3N)+n]
            xₙ₂i = x[2, $N+n] - x[2, $(3N)+n]
        end
        set_lo = quote
            y[1, $(2N)+m] = -yₘ₁r
            y[2, $(2N)+m] = -yₘ₁i
            y[1, $(3N)+m] = -yₘ₂r
            y[2, $(3N)+m] = -yₘ₂i
        end
    else
        return :(throw(DimensionMismatch("ρ must be in [-4,4]")))
    end

    get_Aₘₙ = if is_adjoint === true
        quote
            Aₘₙr = A[1, n, m]
            Aₘₙi = -A[2, n, m]
        end
    else
        quote
            Aₘₙr = A[1, m, n]
            Aₘₙi = A[2, m, n]
        end
    end

    loop_q = quote
        @turbo for m in Base.Slice(static(1):static($N))
            yₘ₁r = $(zero(T))
            yₘ₁i = $(zero(T))
            yₘ₂r = $(zero(T))
            yₘ₂i = $(zero(T))
            for n in Base.Slice(static(1):static($N))
                $get_Aₘₙ
                $calc_hi # @turbo is able to figure out that we only need to do this for m=1 and puts these on the stack
                yₘ₁r += Aₘₙr * xₙ₁r - Aₘₙi * xₙ₁i
                yₘ₁i += Aₘₙr * xₙ₁i + Aₘₙi * xₙ₁r
                yₘ₂r += Aₘₙr * xₙ₂r - Aₘₙi * xₙ₂i
                yₘ₂i += Aₘₙr * xₙ₂i + Aₘₙi * xₙ₂r
            end
            y[1, m] = yₘ₁r
            y[2, m] = yₘ₁i
            y[1, $N+m] = yₘ₂r
            y[2, $N+m] = yₘ₂i
            $set_lo
        end
        return yc
    end

    push!(q.args, loop_q)
    return q
end

"""
    σμν_spin_mul(x, ::Val{μ}, ::Val{ν})

Return `σμν * x` where `σμν = i/2 * [γμ, γν]` with the gamma matrices in the Chiral basis
and `x` is a 4xN component complex vector. The latter two arguments are μ and ν wrapped in
a `Val` and must be within the range `[1,4]` with `μ < ν`
"""
@inline function σμν_spin_mul(x::SVector{M,Complex{T}}, ::Val{μ}, ::Val{ν}) where {T,M,μ,ν}
    return SVector(σμν_spin_mul!(MVector{M,Complex{T}}(undef), MVector(x), Val(μ), Val(ν)))
end

# TODO: make it return two 2Mx2M matrices instead of one 4Mx4M
@generated function σμν_spin_mul!(
    yc::MVector{M,Complex{T}}, xc::MVector{M,Complex{T}}, ::Val{μ}, ::Val{ν}
) where {T,M,μ,ν}
    if M % 4 != 0
        return :(throw(DimensionMismatch("length(x) must be a multiple of 4")))
    end

    N = M ÷ 4

    q = quote
        $(Expr(:meta, :inline))
        y = reinterpret(reshape, $T, yc)
        x = reinterpret(reshape, $T, xc)
    end

    if μ === 1 && ν === 2
        inner_q = quote
            y[1, m] = -x[1, m]
            y[2, m] = -x[2, m]
            y[1, $N+m] = x[1, $N+m]
            y[2, $N+m] = x[2, $N+m]
            y[1, $(2N)+m] = -x[1, $(2N)+m]
            y[2, $(2N)+m] = -x[2, $(2N)+m]
            y[1, $(3N)+m] = x[1, $(3N)+m]
            y[2, $(3N)+m] = x[2, $(3N)+m]
        end
    elseif μ === 1 && ν === 3
        inner_q = quote
            y[1, m] = x[2, $N+m]
            y[2, m] = -x[1, $N+m]
            y[1, $N+m] = -x[2, m]
            y[2, $N+m] = x[1, m]
            y[1, $(2N)+m] = x[2, $(3N)+m]
            y[2, $(2N)+m] = -x[1, $(3N)+m]
            y[1, $(3N)+m] = -x[2, $(2N)+m]
            y[2, $(3N)+m] = x[1, $(2N)+m]
        end
    elseif μ === 1 && ν === 4
        inner_q = quote
            y[1, m] = x[1, $N+m]
            y[2, m] = x[2, $N+m]
            y[1, $N+m] = x[1, m]
            y[2, $N+m] = x[2, m]
            y[1, $(2N)+m] = -x[1, $(3N)+m]
            y[2, $(2N)+m] = -x[2, $(3N)+m]
            y[1, $(3N)+m] = -x[1, $(2N)+m]
            y[2, $(3N)+m] = -x[2, $(2N)+m]
        end
    elseif μ === 2 && ν === 3
        inner_q = quote
            y[1, m] = -x[1, $N+m]
            y[2, m] = -x[2, $N+m]
            y[1, $N+m] = -x[1, m]
            y[2, $N+m] = -x[2, m]
            y[1, $(2N)+m] = -x[1, $(3N)+m]
            y[2, $(2N)+m] = -x[2, $(3N)+m]
            y[1, $(3N)+m] = -x[1, $(2N)+m]
            y[2, $(3N)+m] = -x[2, $(2N)+m]
        end
    elseif μ === 2 && ν === 4
        inner_q = quote
            y[1, m] = x[2, $N+m]
            y[2, m] = -x[1, $N+m]
            y[1, $N+m] = -x[2, m]
            y[2, $N+m] = x[1, m]
            y[1, $(2N)+m] = -x[2, $(3N)+m]
            y[2, $(2N)+m] = x[1, $(3N)+m]
            y[1, $(3N)+m] = x[2, $(2N)+m]
            y[2, $(3N)+m] = -x[1, $(2N)+m]
        end
    elseif μ === 3 && ν === 4
        inner_q = quote
            y[1, m] = x[1, m]
            y[2, m] = x[2, m]
            y[1, $N+m] = -x[1, $N+m]
            y[2, $N+m] = -x[2, $N+m]
            y[1, $(2N)+m] = -x[1, $(2N)+m]
            y[2, $(2N)+m] = -x[2, $(2N)+m]
            y[1, $(3N)+m] = x[1, $(3N)+m]
            y[2, $(3N)+m] = x[2, $(3N)+m]
        end
    else
        return :(throw(DimensionMismatch("Invalid combination of μ and ν")))
    end

    loop_q = quote
        @turbo for m in Base.Slice(static(1):static($N))
            $inner_q
        end
        return yc
    end

    push!(q.args, loop_q)
    return q
end

"""
    cmvmul_pauli(A₊, A₋, x)

Return the matrix-vector product of of the block diagonal matrix made up of and `A₊` and
`A₋` and the vector `x`        
"""
@inline function cmvmul_pauli(
    A₊::SMatrix{N,N,Complex{T},NN},
    A₋::SMatrix{N,N,Complex{T},NN},
    x::SVector{N2,Complex{T}}
) where {T,N,NN,N2}
    x₊ = cmvmul(A₊, SVector{N,Complex{T}}(view(x, 1:N)))
    x₋ = cmvmul(A₊, SVector{N,Complex{T}}(view(x, N+1:2N)))
    return vcat(x₊, x₋)  
end

PrecompileTools.@compile_workload begin
    A64 = @SMatrix rand(ComplexF64, 3, 3)
    v64 = @SVector rand(ComplexF64, 3)
    v164 = @SVector rand(ComplexF64, 12)
    cmvmul(A64, v64)
    cvmmul(v64, A64)
    ckron(v64, v64)
    cmvmul_spin_proj(A64, v164, Val(-1))
    cmvmul_spin_proj(A64, v164, Val(-2))
    cmvmul_spin_proj(A64, v164, Val(-3))
    cmvmul_spin_proj(A64, v164, Val(-4))
    cmvmul_spin_proj(A64, v164, Val(1))
    cmvmul_spin_proj(A64, v164, Val(2))
    cmvmul_spin_proj(A64, v164, Val(3))
    cmvmul_spin_proj(A64, v164, Val(4))
end

using Base.Threads
using LinearAlgebra
using Random
using Polyester
using VectorizedRNG
using StaticArrays

struct SomeKernel1
    state::SVector{nthreads(), Xoshiro}

    function SomeKernel1()
        states = [Xoshiro(sd) for sd in rand(UInt64, nthreads())]
        state = SVector{nthreads(), Xoshiro}(states)
        return new(state)
    end
end

function (sk::SomeKernel1)(U, site, parameter)
    current_value = U[site]
    proposal = sign(current_value) * parameter * current_value^2 + current_value
    accept = (rand(sk.state[threadid()]) < exp(-proposal + current_value))
    accept && (U[site] = proposal)
    return accept
end

function sweep_reduce!(U, count, f!, args...)
    NX, NY, NZ, NT = size(U)
    out = 0.0

    for _ in 1:count
        for pass in 1:2
            @batch per=thread threadlocal=0.0::Float64 for it in 1:NT
                for iz in 1:NZ
                    for iy in 1:NY
                        for ix in 1+mod(it+iz+iy, pass):2:NX
                            site = CartesianIndex(ix, iy, it, iz)
                            threadlocal += f!(U, site, args...)
                        end
                    end
                end
            end
            out += sum(threadlocal)
        end
    end

    return out
end

Random.seed!(123)
U = rand(Float64, 32, 32, 32, 32)
sk = SomeKernel1()
parameter = 0.2

sweep_reduce!(U, 1, sk, parameter)

function testbatch()
    rng = Xoshiro(1206)
    @batch threadlocal=0.0::Float64 for _ in 1:100_000
        threadlocal += rand(rng)
    end
    return sum(threadlocal)
end

# testbatch()

function gaussian_su3_matrix()
    sq3 = sqrt(3)
    h₁ = 0.5 * randn(Float64)
    h₂ = 0.5 * randn(Float64)
    h₃ = 0.5 * randn(Float64)
    h₄ = 0.5 * randn(Float64)
    h₅ = 0.5 * randn(Float64)
    h₆ = 0.5 * randn(Float64)
    h₇ = 0.5 * randn(Float64)
    h₈ = 0.5 * randn(Float64)
    out = @SMatrix [
        im*(h₃+h₈/sq3) h₂+im*h₁        h₅+im*h₄
        -h₂+im*h₁      im*(-h₃+h₈/sq3) h₇+im*h₆
        -h₅+im*h₄      -h₇+im*h₆       im*(-2*h₈/sq3)
    ]
    return out
end

parameters = SVector{4, MVector{8, Float64}}([@MVector zeros(Float64, 8) for _ in 1:nthreads()])
# parameters = [@MVector zeros(Float64, 8) for _ in 1:nthreads()]
function gaussian_su3_matrix(params)
    randn!(local_rng(), params)
    h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈ = 0.5params
    sq3 = sqrt(3)
    out = @SMatrix [
        im*(h₃+h₈/sq3) h₂+im*h₁        h₅+im*h₄
        -h₂+im*h₁      im*(-h₃+h₈/sq3) h₇+im*h₆
        -h₅+im*h₄     -h₇+im*h₆        im*(-2*h₈/sq3)
    ]
    return out
end

function gaussian_momenta_test!(parameters)
    Random.seed!(123)
    VectorizedRNG.seed!(123)
    # out = zeros(Float64, 8nthreads())
    @batch threadlocal=parameters[threadid()]::MVector{8, Float64} for it in 1:8
        for iz in 1:8
            for iy in 1:8
                for ix in 1:8
                    P = gaussian_su3_matrix(threadlocal)
                end
            end
        end
    end
    return sum(sum.(threadlocal))
end

function gaussian_momenta_test1!()
    Random.seed!(123)
    out = zeros(Float64, 8nthreads())
    @batch for it in 1:8
        for iz in 1:8
            for iy in 1:8
                for ix in 1:8
                    P = gaussian_su3_matrix()
                    out += sum(P)
                end
            end
        end
    end
    return sum(sum.(threadlocal))
end

@inline @generated function cmatmul_oo!(
    Cc::MMatrix{NC, NC, Complex{T}, NC2},
    Ac::MMatrix{NC, NC, Complex{T}, NC2},
    Bc::MMatrix{NC, NC, Complex{T}, NC2},
) where {NC, NC2, T}
    quote
        C = reinterpret(reshape, T, Cc)
        A = reinterpret(reshape, T, Ac)
        B = reinterpret(reshape, T, Bc)

        @turbo for n ∈ SOneTo(NC), m ∈ SOneTo(NC)
            Cre = zero(T)
            Cim = zero(T)

            for k ∈ SOneTo(NC)
                Cre += A[1, m, k] * B[1, k, n] - A[2, m, k] * B[2, k, n]
                Cim += A[1, m, k] * B[2, k, n] + A[2, m, k] * B[1, k, n]
            end

            C[1, m, n] = Cre
            C[2, m, n] = Cim
        end
        return Cc
    end
end

@inline @generated function cmatmul_oo(
    A::SMatrix{NC, NC, Complex{T}, NC2},
    B::SMatrix{NC, NC, Complex{T}, NC2},
) where {NC, NC2, T}
    quote
        return SMatrix(cmatmul_oo!(MMatrix{NC, NC, Complex{T}}(undef), MMatrix(A), MMatrix(B)))
    end
end

@generated function testgen(Cc::MMatrix{NC, NC, Complex{T}, NC2}) where {NC, NC2, T}
    quote
        for ii in eachindex(Cc)
            Cc[ii] = 0.1
        end
    end
end

@inline function cmatmul1_oo!(
    Cc::MMatrix{NC, NC, Complex{T}, NC2},
    Ac::MMatrix{NC, NC, Complex{T}, NC2},
    Bc::MMatrix{NC, NC, Complex{T}, NC2},
) where {NC, NC2, T}
    C = reinterpret(reshape, T, Cc)
    A = reinterpret(reshape, T, Ac)
    B = reinterpret(reshape, T, Bc)

    @turbo for n ∈ SOneTo(NC), m ∈ SOneTo(NC)
        Cre = zero(T)
        Cim = zero(T)

        for k ∈ SOneTo(NC)
            Cre += A[1, m, k] * B[1, k, n] - A[2, m, k] * B[2, k, n]
            Cim += A[1, m, k] * B[2, k, n] + A[2, m, k] * B[1, k, n]
        end

        C[1, m, n] = Cre
        C[2, m, n] = Cim
    end

    return Cc
end

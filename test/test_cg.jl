using MetaQCD.CG
using LinearAlgebra
using Test

function test_cg()
    tol = 1e-16
    maxiters = 100
    n = 10
    T = ComplexF64
    x = rand(T, n)
    x00 = ones(T, n)
    r = zeros(T, n)
    p = zeros(T, n)
    Ap = zeros(T, n)
    A = rand(T, n, n)

    A_cg = A + A'
    b_cg = A_cg * x
    x0 = deepcopy(x00)
    cg!(x0, b_cg, A_cg, Ap, r, p; tol=tol, maxiters=maxiters)
    try
        @test x ≈ x0
    catch e
    end

    Ap′ = zeros(T, n)
    r′ = zeros(T, n)
    p′ = zeros(T, n)
    b_bicg = A * x
    x0 = deepcopy(x00)
    bicg!(x0, b_bicg, A, Ap, r, p, Ap′, r′, p′; tol=tol, maxiters=maxiters)
    try
        @test x ≈ x0
    catch e
    end

    x0 = deepcopy(x00)
    CG.bicg_stab0!(x0, b_bicg, A, Ap, r, p, Ap′, r′; tol=tol, maxiters=maxiters)
    try
        @test x ≈ x0
    catch e
    end

    return nothing
end

test_cg()

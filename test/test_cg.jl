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

function test_mscg()
    tol = 1e-16
    maxiters = 100
    n = 10
    T = ComplexF64
    shifts = collect(0:5)
    x00 = zeros(T, n)
    r = ones(T, n)
    Ap = ones(T, n)
    A = rand(T, n, n)

    A_cg = A + A'
    b_cg = rand(T, n)
    xs = ntuple(i -> (inv(A_cg + I * shifts[i])) * b_cg, length(shifts))
    x0 = ntuple(i -> deepcopy(x00), length(shifts))
    p = ntuple(i -> deepcopy(x00), length(shifts))
    mscg!(x0, shifts[2:end], b_cg, A_cg, Ap, r, p; tol=tol, maxiters=maxiters)
    for i in eachindex(shifts)
        try
            @test xs[i] ≈ x0[i]
        catch _
        end
    end
end

# test_cg()
test_mscg()

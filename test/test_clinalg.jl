using MetaQCD.Utils
using LinearAlgebra
using Random
using StaticArrays
using Test

function test_cdot()
    v1s = @SVector rand(ComplexF64, 3)
    v2s = @SVector rand(ComplexF64, 3)
    v3s = @SVector rand(ComplexF64, 12)
    v4s = @SVector rand(ComplexF64, 12)

    @test isapprox(dot(v1s, v2s), cdot(v1s, v2s))
    @test isapprox(dot(v3s, v4s), cdot(v3s, v4s))
end

function test_ckron()
    v1s = @SVector rand(ComplexF64, 3)
    v2s = @SVector rand(ComplexF64, 3)
    v3s = @SVector rand(ComplexF64, 12)
    v4s = @SVector rand(ComplexF64, 12)
    v31s = SVector{3,ComplexF64}(v3s[1:3])
    v32s = SVector{3,ComplexF64}(v3s[4:6])
    v33s = SVector{3,ComplexF64}(v3s[7:9])
    v34s = SVector{3,ComplexF64}(v3s[10:12])
    v41s = SVector{3,ComplexF64}(v4s[1:3])
    v42s = SVector{3,ComplexF64}(v4s[4:6])
    v43s = SVector{3,ComplexF64}(v4s[7:9])
    v44s = SVector{3,ComplexF64}(v4s[10:12])

    @test isapprox(kron(v1s, v2s'), ckron(v1s, v2s))
    @test isapprox(
        kron(v31s, v41s') + kron(v32s, v42s') + kron(v33s, v43s') + kron(v34s, v44s'),
        ckron_sum(v3s, v4s),
    )
end

function test_cmvmul()
    As = @SMatrix rand(ComplexF64, 3, 3)
    vs = @SVector rand(ComplexF64, 3)

    @test isapprox(As * vs, cmvmul(As, vs))
    @test isapprox(As' * vs, cmvmul_d(As, vs))
    @test isapprox(SVector{3,ComplexF64}(vs' * As), cvmmul(vs, As))
    @test isapprox(SVector{3,ComplexF64}(vs' * As'), cvmmul_d(vs, As))
end

function test_spin_color()
    As = @SMatrix rand(ComplexF64, 3, 3)
    vs = @SVector rand(ComplexF64, 12)
    v1s = SVector{3,ComplexF64}(vs[1:3])
    v2s = SVector{3,ComplexF64}(vs[4:6])
    v3s = SVector{3,ComplexF64}(vs[7:9])
    v4s = SVector{3,ComplexF64}(vs[10:12])

    @test isapprox(vcat(As * v1s, As * v2s, As * v3s, As * v4s), cmvmul_color(As, vs))
    @test isapprox(vcat(As' * v1s, As' * v2s, As' * v3s, As' * v4s), cmvmul_d_color(As, vs))
    @test isapprox(
        SVector{12,ComplexF64}(hcat(v1s' * As, v2s' * As, v3s' * As, v4s' * As)),
        cvmmul_color(vs, As),
    )
    @test isapprox(
        SVector{12,ComplexF64}(hcat(v1s' * As', v2s' * As', v3s' * As', v4s' * As')),
        cvmmul_d_color(vs, As),
    )

    # spin_proj
    t1 = (v1s - im * v4s)
    t2 = (v2s - im * v3s)
    onepluγ₁v = vcat(t1, t2, im * t2, im * t1)
    Aonepluγ₁v = vcat(As * t1, As * t2, As * im * t2, As * im * t1)
    t1 = (v1s + im * v4s)
    t2 = (v2s + im * v3s)
    oneminγ₁v = vcat(t1, t2, -im * t2, -im * t1)
    Aoneminγ₁v = vcat(As * t1, As * t2, As * -im * t2, As * -im * t1)
    t1 = (v1s - v4s)
    t2 = (v2s + v3s)
    onepluγ₂v = vcat(t1, t2, t2, -t1)
    Aonepluγ₂v = vcat(As * t1, As * t2, As * t2, As * -t1)
    t1 = (v1s + v4s)
    t2 = (v2s - v3s)
    oneminγ₂v = vcat(t1, t2, -t2, t1)
    Aoneminγ₂v = vcat(As * t1, As * t2, As * -t2, As * t1)
    t1 = (v1s - im * v3s)
    t2 = (v2s + im * v4s)
    onepluγ₃v = vcat(t1, t2, im * t1, -im * t2)
    Aonepluγ₃v = vcat(As * t1, As * t2, As * im * t1, As * -im * t2)
    t1 = (v1s + im * v3s)
    t2 = (v2s - im * v4s)
    oneminγ₃v = vcat(t1, t2, -im * t1, im * t2)
    Aoneminγ₃v = vcat(As * t1, As * t2, As * -im * t1, As * im * t2)
    t1 = (v1s - v3s)
    t2 = (v2s - v4s)
    onepluγ₄v = vcat(t1, t2, -t1, -t2)
    Aonepluγ₄v = vcat(As * t1, As * t2, As * -t1, As * -t2)
    t1 = (v1s + v3s)
    t2 = (v2s + v4s)
    oneminγ₄v = vcat(t1, t2, t1, t2)
    Aoneminγ₄v = vcat(As * t1, As * t2, As * t1, As * t2)

    @test isapprox(onepluγ₁v, spin_proj(vs, Val(1)))
    @test isapprox(oneminγ₁v, spin_proj(vs, Val(-1)))
    @test isapprox(onepluγ₂v, spin_proj(vs, Val(2)))
    @test isapprox(oneminγ₂v, spin_proj(vs, Val(-2)))
    @test isapprox(onepluγ₃v, spin_proj(vs, Val(3)))
    @test isapprox(oneminγ₃v, spin_proj(vs, Val(-3)))
    @test isapprox(onepluγ₄v, spin_proj(vs, Val(4)))
    @test isapprox(oneminγ₄v, spin_proj(vs, Val(-4)))
    @test isapprox(Aonepluγ₁v, cmvmul_spin_proj(As, vs, Val(1), Val(false)))
    @test isapprox(Aoneminγ₁v, cmvmul_spin_proj(As, vs, Val(-1), Val(false)))
    @test isapprox(Aonepluγ₂v, cmvmul_spin_proj(As, vs, Val(2), Val(false)))
    @test isapprox(Aoneminγ₂v, cmvmul_spin_proj(As, vs, Val(-2), Val(false)))
    @test isapprox(Aonepluγ₃v, cmvmul_spin_proj(As, vs, Val(3), Val(false)))
    @test isapprox(Aoneminγ₃v, cmvmul_spin_proj(As, vs, Val(-3), Val(false)))
    @test isapprox(Aonepluγ₄v, cmvmul_spin_proj(As, vs, Val(4), Val(false)))
    @test isapprox(Aoneminγ₄v, cmvmul_spin_proj(As, vs, Val(-4), Val(false)))
end

function test_cmatmul()
    Random.seed!(123)
    As = @SMatrix rand(ComplexF64, 3, 3)
    Bs = @SMatrix rand(ComplexF64, 3, 3)
    Cs = @SMatrix rand(ComplexF64, 3, 3)
    Ds = @SMatrix rand(ComplexF64, 3, 3)

    @test isapprox(As * Bs, cmatmul_oo(As, Bs))
    @test isapprox(As * Bs', cmatmul_od(As, Bs))
    @test isapprox(As' * Bs, cmatmul_do(As, Bs))
    @test isapprox(As' * Bs', cmatmul_dd(As, Bs))

    @test isapprox(As * Bs * Cs, cmatmul_ooo(As, Bs, Cs))
    @test isapprox(As * Bs * Cs', cmatmul_ood(As, Bs, Cs))
    @test isapprox(As * Bs' * Cs, cmatmul_odo(As, Bs, Cs))
    @test isapprox(As' * Bs * Cs, cmatmul_doo(As, Bs, Cs))
    @test isapprox(As * Bs' * Cs', cmatmul_odd(As, Bs, Cs))
    @test isapprox(As' * Bs' * Cs, cmatmul_ddo(As, Bs, Cs))
    @test isapprox(As' * Bs * Cs', cmatmul_dod(As, Bs, Cs))
    @test isapprox(As' * Bs' * Cs', cmatmul_ddd(As, Bs, Cs))

    @test isapprox(As * Bs * Cs * Ds, cmatmul_oooo(As, Bs, Cs, Ds))
    @test isapprox(As * Bs * Cs * Ds', cmatmul_oood(As, Bs, Cs, Ds))
    @test isapprox(As * Bs * Cs' * Ds, cmatmul_oodo(As, Bs, Cs, Ds))
    @test isapprox(As * Bs' * Cs * Ds, cmatmul_odoo(As, Bs, Cs, Ds))
    @test isapprox(As' * Bs * Cs * Ds, cmatmul_dooo(As, Bs, Cs, Ds))
    @test isapprox(As * Bs * Cs' * Ds', cmatmul_oodd(As, Bs, Cs, Ds))
    @test isapprox(As * Bs' * Cs' * Ds, cmatmul_oddo(As, Bs, Cs, Ds))
    @test isapprox(As' * Bs' * Cs * Ds, cmatmul_ddoo(As, Bs, Cs, Ds))
    @test isapprox(As * Bs' * Cs * Ds', cmatmul_odod(As, Bs, Cs, Ds))
    @test isapprox(As' * Bs * Cs * Ds', cmatmul_dood(As, Bs, Cs, Ds))
    @test isapprox(As' * Bs * Cs' * Ds, cmatmul_dodo(As, Bs, Cs, Ds))
    @test isapprox(As * Bs' * Cs' * Ds', cmatmul_oddd(As, Bs, Cs, Ds))
    @test isapprox(As' * Bs' * Cs' * Ds, cmatmul_dddo(As, Bs, Cs, Ds))
    @test isapprox(As' * Bs' * Cs * Ds', cmatmul_ddod(As, Bs, Cs, Ds))
    @test isapprox(As' * Bs * Cs' * Ds', cmatmul_dodd(As, Bs, Cs, Ds))
    @test isapprox(As' * Bs' * Cs' * Ds', cmatmul_dddd(As, Bs, Cs, Ds))
end

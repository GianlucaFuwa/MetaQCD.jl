using MetaQCD.Utils
using Random
using StaticArrays

function SU3testlinalg()
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

# using LoopVectorization
# using Random
# using StaticArrays
# using Test
#========== 2-fold products ==========#
@inline function cmatmul_oo!(
    Cc::MMatrix{NC, NC, ComplexF64, NC2},
    Ac::MMatrix{NC, NC, ComplexF64, NC2},
    Bc::MMatrix{NC, NC, ComplexF64, NC2},
) where {NC, NC2}
    C = reinterpret(reshape, Float64, Cc)
    A = reinterpret(reshape, Float64, Ac)
    B = reinterpret(reshape, Float64, Bc)

    @turbo for n ∈ Base.Slice(static(1):static(NC)), m ∈ Base.Slice(static(1):static(NC))
        Cre = zero(Float64)
        Cim = zero(Float64)

        for k ∈ Base.Slice(static(1):static(NC))
            Cre += A[1, m, k] * B[1, k, n] - A[2, m, k] * B[2, k, n]
            Cim += A[1, m, k] * B[2, k, n] + A[2, m, k] * B[1, k, n]
        end

        C[1, m, n] = Cre
        C[2, m, n] = Cim
    end

    return Cc
end

@inline function cmatmul_oo(A::SMatrix, B::SMatrix)
    return SMatrix(cmatmul_oo!(MMatrix{3, 3, ComplexF64}(undef), MMatrix(A), MMatrix(B)))
end

@inline function cmatmul_od!(
    Cc::MMatrix{NC, NC, ComplexF64, NC2},
    Ac::MMatrix{NC, NC, ComplexF64, NC2},
    Bc::MMatrix{NC, NC, ComplexF64, NC2},
) where {NC, NC2}
    C = reinterpret(reshape, Float64, Cc)
    A = reinterpret(reshape, Float64, Ac)
    B = reinterpret(reshape, Float64, Bc)

    @turbo for n ∈ Base.Slice(static(1):static(NC)), m ∈ Base.Slice(static(1):static(NC))
        Cre = zero(Float64)
        Cim = zero(Float64)

        for k ∈ Base.Slice(static(1):static(NC))
            Cre += A[1, m, k] * B[1, n, k] + A[2, m, k] * B[2, n, k]
            Cim += -A[1, m, k] * B[2, n, k] + A[2, m, k] * B[1, n, k]
        end

        C[1, m, n] = Cre
        C[2, m, n] = Cim
    end

    return Cc
end

@inline function cmatmul_od(A::SMatrix, B::SMatrix)
    return SMatrix(cmatmul_od!(MMatrix{3, 3, ComplexF64}(undef), MMatrix(A), MMatrix(B)))
end

@inline function cmatmul_do!(
    Cc::MMatrix{NC, NC, ComplexF64, NC2},
    Ac::MMatrix{NC, NC, ComplexF64, NC2},
    Bc::MMatrix{NC, NC, ComplexF64, NC2},
) where {NC, NC2}
    C = reinterpret(reshape, Float64, Cc)
    A = reinterpret(reshape, Float64, Ac)
    B = reinterpret(reshape, Float64, Bc)

    @turbo for n ∈ Base.Slice(static(1):static(NC)), m ∈ Base.Slice(static(1):static(NC))
        Cre = zero(Float64)
        Cim = zero(Float64)

        for k ∈ Base.Slice(static(1):static(NC))
            Cre += A[1, k, m] * B[1, k, n] + A[2, k, m] * B[2, k, n]
            Cim += A[1, k, m] * B[2, k, n] - A[2, k, m] * B[1, k, n]
        end

        C[1, m, n] = Cre
        C[2, m, n] = Cim
    end

    return Cc
end

@inline function cmatmul_do(A::SMatrix, B::SMatrix)
    return SMatrix(cmatmul_do!(MMatrix{3, 3, ComplexF64}(undef), MMatrix(A), MMatrix(B)))
end

@inline function cmatmul_dd!(
    Cc::MMatrix{NC, NC, ComplexF64, NC2},
    Ac::MMatrix{NC, NC, ComplexF64, NC2},
    Bc::MMatrix{NC, NC, ComplexF64, NC2},
) where {NC, NC2}
    C = reinterpret(reshape, Float64, Cc)
    A = reinterpret(reshape, Float64, Ac)
    B = reinterpret(reshape, Float64, Bc)

    @turbo for n ∈ Base.Slice(static(1):static(NC)), m ∈ Base.Slice(static(1):static(NC))
        Cre = zero(Float64)
        Cim = zero(Float64)

        for k ∈ Base.Slice(static(1):static(NC))
            Cre += A[1, k, m] * B[1, n, k] - A[2, k, m] * B[2, n, k]
            Cim += -A[1, k, m] * B[2, n, k] - A[2, k, m] * B[1, n, k]
        end

        C[1, m, n] = Cre
        C[2, m, n] = Cim
    end

    return Cc
end

@inline function cmatmul_dd(A::SMatrix, B::SMatrix)
    return SMatrix(cmatmul_dd!(MMatrix{3, 3, ComplexF64}(undef), MMatrix(A), MMatrix(B)))
end

#========== 3-fold products ==========#
@inline function cmatmul_ooo!(Dc, Ac, Bc, Cc)
    cmatmul_oo!(Dc, Ac, cmatmul_oo!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_ooo(A::SMatrix, B::SMatrix, C::SMatrix)
    return SMatrix(cmatmul_ooo!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C)
    ))
end

@inline function cmatmul_ood!(Dc, Ac, Bc, Cc)
    cmatmul_oo!(Dc, Ac, cmatmul_od!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_ood(A::SMatrix, B::SMatrix, C::SMatrix)
    return SMatrix(cmatmul_ood!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C)
    ))
end

@inline function cmatmul_odo!(Dc, Ac, Bc, Cc)
    cmatmul_oo!(Dc, Ac, cmatmul_do!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_odo(A::SMatrix, B::SMatrix, C::SMatrix)
    return SMatrix(cmatmul_odo!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C)
    ))
end

@inline function cmatmul_doo!(Dc, Ac, Bc, Cc)
    cmatmul_do!(Dc, Ac, cmatmul_oo!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_doo(A::SMatrix, B::SMatrix, C::SMatrix)
    return SMatrix(cmatmul_doo!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C)
    ))
end

@inline function cmatmul_odd!(Dc, Ac, Bc, Cc)
    cmatmul_oo!(Dc, Ac, cmatmul_dd!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_odd(A::SMatrix, B::SMatrix, C::SMatrix)
    return SMatrix(cmatmul_odd!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C)
    ))
end

@inline function cmatmul_ddo!(Dc, Ac, Bc, Cc)
    cmatmul_do!(Dc, Ac, cmatmul_do!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_ddo(A::SMatrix, B::SMatrix, C::SMatrix)
    return SMatrix(cmatmul_ddo!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C)
    ))
end

@inline function cmatmul_dod!(Dc, Ac, Bc, Cc)
    cmatmul_do!(Dc, Ac, cmatmul_od!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_dod(A::SMatrix, B::SMatrix, C::SMatrix)
    return SMatrix(cmatmul_dod!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C)
    ))
end

@inline function cmatmul_ddd!(Dc, Ac, Bc, Cc)
    cmatmul_do!(Dc, Ac, cmatmul_dd!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_ddd(A::SMatrix, B::SMatrix, C::SMatrix)
    return SMatrix(cmatmul_ddd!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C)
    ))
end

#========== 4-fold products ==========#
@inline function cmatmul_oooo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_ooo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_oooo(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_oooo!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_oood!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_ood!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_oood(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_oood!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_oodo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_odo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_oodo(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_oodo!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_odoo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_doo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_odoo(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_odoo!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_dooo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_ooo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_dooo(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_dooo!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_oodd!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_odd!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_oodd(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_oodd!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_oddo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_ddo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_oddo(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_oddo!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_ddoo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_doo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_ddoo(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_ddoo!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_dood!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_ood!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_dood(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_dood!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_odod!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_dod!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_odod(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_odod!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_dodo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_odo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_dodo(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_dodo!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_oddd!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_ddd!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_oddd(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_oddd!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_dodd!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_odd!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_dodd(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_dodd!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_ddod!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_dod!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_ddod(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_ddod!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_dddo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_ddo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_dddo(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_dddo!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

@inline function cmatmul_dddd!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_ddd!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_dddd(A::SMatrix, B::SMatrix, C::SMatrix, D::SMatrix)
    return SMatrix(cmatmul_dddd!(
        MMatrix{3,3,ComplexF64}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
    ))
end

# @testset "matmul" begin
#     Random.seed!(1206)
#     As = @SMatrix rand(ComplexF64, 3, 3)
#     Bs = @SMatrix rand(ComplexF64, 3, 3)
#     Cs = @SMatrix rand(ComplexF64, 3, 3)
#     Ds = @SMatrix rand(ComplexF64, 3, 3)

#     @test isapprox(As  * Bs , cmatmul_oo(As, Bs))
#     @test isapprox(As  * Bs', cmatmul_od(As, Bs))
#     @test isapprox(As' * Bs , cmatmul_do(As, Bs))
#     @test isapprox(As' * Bs', cmatmul_dd(As, Bs))

#     @test isapprox(As  * Bs  * Cs , cmatmul_ooo(As, Bs, Cs))
#     @test isapprox(As  * Bs  * Cs', cmatmul_ood(As, Bs, Cs))
#     @test isapprox(As  * Bs' * Cs , cmatmul_odo(As, Bs, Cs))
#     @test isapprox(As' * Bs  * Cs , cmatmul_doo(As, Bs, Cs))
#     @test isapprox(As  * Bs' * Cs', cmatmul_odd(As, Bs, Cs))
#     @test isapprox(As' * Bs' * Cs , cmatmul_ddo(As, Bs, Cs))
#     @test isapprox(As' * Bs  * Cs', cmatmul_dod(As, Bs, Cs))
#     @test isapprox(As' * Bs' * Cs', cmatmul_ddd(As, Bs, Cs))

#     @test isapprox(As  * Bs  * Cs  * Ds , cmatmul_oooo(As, Bs, Cs, Ds))
#     @test isapprox(As  * Bs  * Cs  * Ds', cmatmul_oood(As, Bs, Cs, Ds))
#     @test isapprox(As  * Bs  * Cs' * Ds , cmatmul_oodo(As, Bs, Cs, Ds))
#     @test isapprox(As  * Bs' * Cs  * Ds , cmatmul_odoo(As, Bs, Cs, Ds))
#     @test isapprox(As' * Bs  * Cs  * Ds , cmatmul_dooo(As, Bs, Cs, Ds))
#     @test isapprox(As  * Bs  * Cs' * Ds', cmatmul_oodd(As, Bs, Cs, Ds))
#     @test isapprox(As  * Bs' * Cs' * Ds , cmatmul_oddo(As, Bs, Cs, Ds))
#     @test isapprox(As' * Bs' * Cs  * Ds , cmatmul_ddoo(As, Bs, Cs, Ds))
#     @test isapprox(As  * Bs' * Cs  * Ds', cmatmul_odod(As, Bs, Cs, Ds))
#     @test isapprox(As' * Bs  * Cs  * Ds', cmatmul_dood(As, Bs, Cs, Ds))
#     @test isapprox(As' * Bs  * Cs' * Ds , cmatmul_dodo(As, Bs, Cs, Ds))
#     @test isapprox(As  * Bs' * Cs' * Ds', cmatmul_oddd(As, Bs, Cs, Ds))
#     @test isapprox(As' * Bs' * Cs' * Ds , cmatmul_dddo(As, Bs, Cs, Ds))
#     @test isapprox(As' * Bs' * Cs  * Ds', cmatmul_ddod(As, Bs, Cs, Ds))
#     @test isapprox(As' * Bs  * Cs' * Ds', cmatmul_dodd(As, Bs, Cs, Ds))
#     @test isapprox(As' * Bs' * Cs' * Ds', cmatmul_dddd(As, Bs, Cs, Ds))
# end

@inline function cmatmul_oo!(
    Cc::MMatrix{NC,NC,Complex{T},NC2},
    Ac::MMatrix{NC,NC,Complex{T},NC2},
    Bc::MMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    C = reinterpret(reshape, T, Cc)
    A = reinterpret(reshape, T, Ac)
    B = reinterpret(reshape, T, Bc)

    @turbo for n in Base.Slice(static(1):static(NC)), m in Base.Slice(static(1):static(NC))
        Cre = zero(T)
        Cim = zero(T)

        for k in Base.Slice(static(1):static(NC))
            Cre += A[1, m, k] * B[1, k, n] - A[2, m, k] * B[2, k, n]
            Cim += A[1, m, k] * B[2, k, n] + A[2, m, k] * B[1, k, n]
        end

        C[1, m, n] = Cre
        C[2, m, n] = Cim
    end

    return Cc
end

@inline function cmatmul_oo(
    A::SMatrix{NC,NC,Complex{T},NC2}, B::SMatrix{NC,NC,Complex{T},NC2}
) where {NC,NC2,T}
    return SMatrix(cmatmul_oo!(MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B)))
end

@inline function cmatmul_od!(
    Cc::MMatrix{NC,NC,Complex{T},NC2},
    Ac::MMatrix{NC,NC,Complex{T},NC2},
    Bc::MMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    C = reinterpret(reshape, T, Cc)
    A = reinterpret(reshape, T, Ac)
    B = reinterpret(reshape, T, Bc)

    @turbo for n in Base.Slice(static(1):static(NC)), m in Base.Slice(static(1):static(NC))
        Cre = zero(T)
        Cim = zero(T)

        for k in Base.Slice(static(1):static(NC))
            Cre += A[1, m, k] * B[1, n, k] + A[2, m, k] * B[2, n, k]
            Cim += -A[1, m, k] * B[2, n, k] + A[2, m, k] * B[1, n, k]
        end

        C[1, m, n] = Cre
        C[2, m, n] = Cim
    end

    return Cc
end

@inline function cmatmul_od(
    A::SMatrix{NC,NC,Complex{T},NC2}, B::SMatrix{NC,NC,Complex{T},NC2}
) where {NC,NC2,T}
    return SMatrix(cmatmul_od!(MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B)))
end

@inline function cmatmul_do!(
    Cc::MMatrix{NC,NC,Complex{T},NC2},
    Ac::MMatrix{NC,NC,Complex{T},NC2},
    Bc::MMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    C = reinterpret(reshape, T, Cc)
    A = reinterpret(reshape, T, Ac)
    B = reinterpret(reshape, T, Bc)

    @turbo for n in Base.Slice(static(1):static(NC)), m in Base.Slice(static(1):static(NC))
        Cre = zero(T)
        Cim = zero(T)

        for k in Base.Slice(static(1):static(NC))
            Cre += A[1, k, m] * B[1, k, n] + A[2, k, m] * B[2, k, n]
            Cim += A[1, k, m] * B[2, k, n] - A[2, k, m] * B[1, k, n]
        end

        C[1, m, n] = Cre
        C[2, m, n] = Cim
    end

    return Cc
end

@inline function cmatmul_do(
    A::SMatrix{NC,NC,Complex{T},NC2}, B::SMatrix{NC,NC,Complex{T},NC2}
) where {NC,NC2,T}
    return SMatrix(cmatmul_do!(MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B)))
end

@inline function cmatmul_dd!(
    Cc::MMatrix{NC,NC,Complex{T},NC2},
    Ac::MMatrix{NC,NC,Complex{T},NC2},
    Bc::MMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    C = reinterpret(reshape, T, Cc)
    A = reinterpret(reshape, T, Ac)
    B = reinterpret(reshape, T, Bc)

    @turbo for n in Base.Slice(static(1):static(NC)), m in Base.Slice(static(1):static(NC))
        Cre = zero(T)
        Cim = zero(T)

        for k in Base.Slice(static(1):static(NC))
            Cre += A[1, k, m] * B[1, n, k] - A[2, k, m] * B[2, n, k]
            Cim += -A[1, k, m] * B[2, n, k] - A[2, k, m] * B[1, n, k]
        end

        C[1, m, n] = Cre
        C[2, m, n] = Cim
    end

    return Cc
end

@inline function cmatmul_dd(
    A::SMatrix{NC,NC,Complex{T},NC2}, B::SMatrix{NC,NC,Complex{T},NC2}
) where {NC,NC2,T}
    return SMatrix(cmatmul_dd!(MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B)))
end

@inline function cmatmul_ooo!(Dc, Ac, Bc, Cc)
    cmatmul_oo!(Dc, Ac, cmatmul_oo!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_ooo(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_ooo!(MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C))
    )
end

@inline function cmatmul_ood!(Dc, Ac, Bc, Cc)
    cmatmul_oo!(Dc, Ac, cmatmul_od!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_ood(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_ood!(MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C))
    )
end

@inline function cmatmul_odo!(Dc, Ac, Bc, Cc)
    cmatmul_oo!(Dc, Ac, cmatmul_do!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_odo(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_odo!(MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C))
    )
end

@inline function cmatmul_doo!(Dc, Ac, Bc, Cc)
    cmatmul_do!(Dc, Ac, cmatmul_oo!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_doo(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_doo!(MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C))
    )
end

@inline function cmatmul_odd!(Dc, Ac, Bc, Cc)
    cmatmul_oo!(Dc, Ac, cmatmul_dd!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_odd(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_odd!(MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C))
    )
end

@inline function cmatmul_ddo!(Dc, Ac, Bc, Cc)
    cmatmul_do!(Dc, Ac, cmatmul_do!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_ddo(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_ddo!(MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C))
    )
end

@inline function cmatmul_dod!(Dc, Ac, Bc, Cc)
    cmatmul_do!(Dc, Ac, cmatmul_od!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_dod(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_dod!(MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C))
    )
end

@inline function cmatmul_ddd!(Dc, Ac, Bc, Cc)
    cmatmul_do!(Dc, Ac, cmatmul_dd!(Dc, Bc, Cc))
    return Dc
end

@inline function cmatmul_ddd(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_ddd!(MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C))
    )
end

@inline function cmatmul_oooo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_ooo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_oooo(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_oooo!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_oood!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_ood!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_oood(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_oood!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_oodo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_odo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_oodo(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_oodo!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_odoo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_doo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_odoo(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_odoo!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_dooo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_ooo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_dooo(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_dooo!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_oodd!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_odd!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_oodd(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_oodd!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_oddo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_ddo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_oddo(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_oddo!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_ddoo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_doo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_ddoo(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_ddoo!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_dood!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_ood!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_dood(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_dood!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_odod!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_dod!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_odod(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_odod!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_dodo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_odo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_dodo(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_dodo!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_oddd!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_oo!(Ec, Ac, cmatmul_ddd!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_oddd(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_oddd!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_dodd!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_odd!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_dodd(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_dodd!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_ddod!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_dod!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_ddod(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_ddod!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_dddo!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_ddo!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_dddo(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_dddo!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

@inline function cmatmul_dddd!(Ec, Ac, Bc, Cc, Dc)
    cmatmul_do!(Ec, Ac, cmatmul_ddd!(Ec, Bc, Cc, Dc))
    return Ec
end

@inline function cmatmul_dddd(
    A::SMatrix{NC,NC,Complex{T},NC2},
    B::SMatrix{NC,NC,Complex{T},NC2},
    C::SMatrix{NC,NC,Complex{T},NC2},
    D::SMatrix{NC,NC,Complex{T},NC2},
) where {NC,NC2,T}
    return SMatrix(
        cmatmul_dddd!(
            MMatrix{NC,NC,Complex{T}}(undef), MMatrix(A), MMatrix(B), MMatrix(C), MMatrix(D)
        ),
    )
end

# Do some precompilation to reduce time to first execution (TTFX)
PrecompileTools.@compile_workload begin
    A64 = @SMatrix rand(ComplexF64, 3, 3)
    B64 = @SMatrix rand(ComplexF64, 3, 3)
    C64 = @SMatrix rand(ComplexF64, 3, 3)
    D64 = @SMatrix rand(ComplexF64, 3, 3)
    cmatmul_oood(A64, B64, C64, D64)
    cmatmul_dood(A64, B64, C64, D64)
end

    
    

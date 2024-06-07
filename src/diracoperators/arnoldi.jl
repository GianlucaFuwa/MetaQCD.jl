import ArnoldiMethod: partialschur!, get_order
using ArnoldiMethod
 
struct ArnoldiWorkspaceMeta{T,TV,K,K1,KK,K1K}
    V::TV
    V_tmp::TV
    H::MMatrix{K1,K,Complex{T},K1K}
    Q::MMatrix{K,K,Complex{T},KK}
    h::MVector{K1,Complex{T}}
    x::MVector{K,Complex{T}}
    groups::MVector{K,Int}
    G::ArnoldiMethod.Reflector{Complex{T}}
    ritz::ArnoldiMethod.RitzValues{Complex{T},T}
    function ArnoldiWorkspaceMeta(D::AbstractDiracOperator, k)
        temp = get_temp(D)
        T = eltype(temp)
        V = [similar(temp) for _ in 1:k+1]
        V_tmp = [similar(temp) for _ in 1:k+1]
        H = @MMatrix zeros(T, k+1, k)
        Q = @MMatrix zeros(T, k, k)
        h = @MVector zeros(T, k+1)
        x = @MVector zeros(T, k)
        groups = @MVector zeros(Int, k)
        G = ArnoldiMethod.Reflector{T}(k)
        ritz = ArnoldiMethod.RitzValues{T}(k)

        TV = typeof(V)
        K = k
        K1 = k + 1
        KK = k * k
        K1K = (k + 1) * k
        return new{real(T),TV,K,K1,KK,K1K}(V, V_tmp, H, Q, h, x, groups, G, ritz)
    end
end

function get_eigenvalues(
    U::Gaugefield,
    D::AbstractDiracOperator;
    nev::Int = 1,
    which::Symbol = :LM,
    tol::Real = sqrt(eps(Float64)),
    mindim::Int = max(10, nev),
    maxdim::Int = max(20, 2nev),
    restarts::Int = 200,
    ddaggerd::Bool = false,
)
    arnoldi = ArnoldiWorkspaceMeta(D, maxdim)
    D_in = ddaggerd ? DdaggerD(D(U)) : D(U)
    schur, _ = partialschur!(
        D_in,
        arnoldi;
        nev=nev,
        which=which,
        tol=tol,
        mindim=mindim,
        restarts=restarts,
    )
    return partialeigen_meta(schur)
end

function get_eigenvalues(
    U::Gaugefield,
    D::AbstractDiracOperator,
    arnoldi::ArnoldiWorkspaceMeta{T};
    nev::Int = 1,
    which::Symbol = :LM,
    tol::Real = sqrt(eps(real(T))),
    mindim::Int = min(max(10, nev), length(arnoldi.V)-1),
    restarts::Int = 200,
    ddaggerd::Bool = false,
) where {T}
    D_in = ddaggerd ? DdaggerD(D(U)) : D(U)
    arnoldi.H .= 0
    arnoldi.Q .= 0
    arnoldi.h .= 0
    arnoldi.x .= 0
    schur, _ = partialschur!(
        D_in,
        arnoldi;
        nev=nev,
        which=which,
        tol=tol,
        mindim=mindim,
        restarts=restarts,
    )
    return partialeigen_meta(schur)
end

function partialeigen_meta(P::ArnoldiMethod.PartialSchur; arnoldi=nothing)
    vals, vecs = eigen(P.R)
    
    if arnoldi isa ArnoldiWorkspaceMeta
        vecs
        # Store eigenvectors
    end
    return vals
end

function partialschur!(
    D::AbstractDiracOperator,
    arnoldi::ArnoldiWorkspaceMeta{T};
    nev::Int = 1,
    which::Symbol = :LM,
    tol::Real = sqrt(eps(real(T))),
    mindim::Int = min(max(10, nev), length(arnoldi.x)),
    restarts::Int = 200,
) where {T}
    n = LinearAlgebra.checksquare(D)
    maxdim = length(arnoldi.x)
    @assert nev ≥ 1 "nev cannot be less than 1, was $nev"
    @assert nev ≤ mindim ≤ maxdim < n "nev ≤ mindim ≤ maxdim < n must hold, was $nev ≤ $mindim ≤ $maxdim < $n"
    _which = _symbol_to_target_meta(which)
    fill!(view(arnoldi.H, :, axes(arnoldi.H, 2)), zero(T)) 
    reinitialize_meta!(arnoldi, 0)
    _partialschur(D, arnoldi, mindim, maxdim, nev, tol, restarts, _which)
end

function _partialschur(D, arnoldi, mindim, maxdim, nev, tol, restarts, which)
    H = arnoldi.H
    V = arnoldi.V
    V_tmp = arnoldi.V_tmp
    Q = arnoldi.Q
    x = arnoldi.x
    groups = arnoldi.groups
    G = arnoldi.G
    ritz = arnoldi.ritz

    T = eltype(H)
    isconverged = ArnoldiMethod.IsConverged(ritz, tol)
    ordering = get_order(which)

    active = 1
    k = mindim
    effective_nev = nev
    prods = mindim

    iterate_arnoldi_meta!(D, arnoldi, 1:mindim)

    for _ in 1:restarts
        iterate_arnoldi_meta!(D, arnoldi, k+1:maxdim)
        
        prods += length(k+1:maxdim)

        copyto!(Q, I)

        ArnoldiMethod.local_schurfact!(view(H, Base.OneTo(maxdim), :), 1, maxdim, Q)

        copyto!(ritz.ord, Base.OneTo(maxdim))
        ArnoldiMethod.copy_eigenvalues!(ritz.λs, H)
        ArnoldiMethod.copy_residuals!(ritz.rs, H, Q, H[maxdim+1, maxdim], x, 1:maxdim)

        sort!(ritz.ord, 1, maxdim, QuickSort, ArnoldiMethod.OrderPerm(ritz.λs, ordering))

        isconverged.H_frob_norm[] = norm(H)

        effective_nev = ArnoldiMethod.include_conjugate_pair(T, ritz, nev)

        nlock = 0
        @inbounds for i in 1:effective_nev
            if isconverged(ritz.ord[i])
                groups[ritz.ord[i]] = 1
                nlock += 1
            else
                groups[ritz.ord[i]] = 2
            end
        end

        ideal_size = min(nlock + mindim, (mindim + maxdim) ÷ 2)
        k = effective_nev
        i = effective_nev + 1

        @inbounds while i ≤ maxdim
            is_pair = ArnoldiMethod.include_conjugate_pair(T, ritz, i) == i + 1
            num = is_pair ? 2 : 1
            if k < ideal_size && !isconverged(ritz.ord[i])
                group = 2
                k += num
            else
                group = 3
            end
            if is_pair
                groups[ritz.ord[i]] = group
                groups[ritz.ord[i+1]] = group
                i += 2
            else
                groups[ritz.ord[i]] = group
                i += 1
            end
        end

        purge = 1
        @inbounds while purge < active && groups[purge] == 1
            purge += 1
        end

        ArnoldiMethod.partition_schur_three_way!(H, Q, groups)
        ArnoldiMethod.restore_arnoldi!(H, nlock + 1, k, Q, G)

        for i in purge:k
            clear!(V_tmp[i])
            for j in purge:maxdim
                axpy!(Q[j, i], V[j], V_tmp[i])
            end
        end
        for i in purge:k
            copy!(V[i], V_tmp[i])
        end
        copy!(V[k+1], V[maxdim+1])
        
        active = nlock + 1
        active > nev && break
    end

    nconverged = active - 1

    @views Vconverged = V[1:nconverged]
    @views Hconverged = H[1:nconverged, 1:nconverged]
    
    ArnoldiMethod.sortschur!(H, copyto!(Q, I), nconverged, ordering)

    # Change of basis
    for i in 1:nconverged
        clear!(V_tmp[i])
        for j in 1:nconverged
            axpy!(Q[j, i], Vconverged[j], V_tmp[i]) 
        end
    end
    for i in 1:nconverged
        copy!(Vconverged[i], V_tmp[i])
    end
    
    ArnoldiMethod.copy_eigenvalues!(ritz.λs, H, Base.OneTo(nconverged))
    history = ArnoldiMethod.History(prods, nconverged, nconverged ≥ nev, nev)
    schur = ArnoldiMethod.PartialSchur(Vconverged, Hconverged, ritz.λs[1:nconverged])

    return schur, history
end

function reinitialize_meta!(arnoldi, j)
    V = arnoldi.V
    h = arnoldi.h
    v = V[j+1]
    gaussian_pseudofermions!(v)
    rnorm = norm(v)

    if j==0
        mul!(v, 1 / rnorm)
        return true
    end

    η = sqrt(2) / 2

    for i in 1:j
        h[i] = dot(V[i], v)
    end
    for i in 1:j
        axpy!(-h[i], V[i], v)
    end

    wnorm = norm(v)

    if wnorm < η * rnorm
        rnorm = wnorm
        for i in 1:j
            h[i] = dot(V[i], v)
        end
        for i in 1:j
            axpy!(-h[i], V[i], v)
        end
        wnorm = norm(v)
    end

    if wnorm ≤ η * rnorm
        return false
    else
        mul!(v, 1 / wnorm)
        return true
    end
end

function orthogonalize_meta!(arnoldi, j)
    V, H = arnoldi.V, arnoldi.H
    correction = @views arnoldi.x[1:j]

    η = sqrt(2) / 2
    v = V[j+1]
    h = view(H, 1:j, j)

    rnorm = norm(v)

    for i in 1:j
        h[i] = dot(V[i], v)
    end

    for i in 1:j
        axpy!(-h[i], V[i], v)
    end

    wnorm = norm(v)

    if wnorm < η * rnorm
        rnorm = wnorm
        for i in 1:j
            correction[i] = dot(V[i], v)
        end
        for i in 1:j
            axpy!(-correction[i], V[i], v)
        end
        h .+= correction
        wnorm = norm(v)
    end

    if wnorm ≤ η * rnorm
        H[j+1, j] = 0
        return false
    else
        H[j+1, j] = wnorm
        mul!(v, 1 / wnorm)
        return true
    end
end

function iterate_arnoldi_meta!(D, arnoldi, range)
    V = arnoldi.V

    for j in range
        mul!(V[j+1], D, V[j])

        if orthogonalize_meta!(arnoldi, j) == false && j != length(V)
            reinitialize_meta!(arnoldi, j)
        end
    end

    return nothing
end

# Because ArnoldiMethod doesn't support smallest absolute value...
struct SM <: ArnoldiMethod.Target end

_symbol_to_target_meta(sym::Symbol) =
    sym == :LM ? ArnoldiMethod.LM() :
    sym == :SM ? SM() :
    sym == :LR ? ArnoldiMethod.LR() :
    sym == :SR ? ArnoldiMethod.SR() :
    sym == :LI ? ArnoldiMethod.LI() :
    sym == :SI ? ArnoldiMethod.SI() : throw(ArgumentError("Unknown target: $sym"))

get_order(::SM) = ArnoldiMethod.OrderBy(abs)

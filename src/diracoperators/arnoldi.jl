import ArnoldiMethod: partialschur!, get_order
using ArnoldiMethod
 
struct ArnoldiWorkspaceMeta{T,TV,TH,TVtmp,TQ}
    V::TV
    H::TH
    V_tmp::TVtmp
    Q::TQ
end

function ArnoldiWorkspaceMeta(D::AbstractDiracOperator, k)
    temp = get_temp(D)
    V = ntuple(_ -> similar(temp), k+1)
    V_tmp = ntuple(_ -> similar(temp), k+1)
    H = zeros(eltype(temp), k+1, k)
    Q = zeros(eltype(temp), k, k)

    TV = typeof(V)
    TH = typeof(H)
    TVtmp = typeof(V_tmp)
    TQ = typeof(Q)
    return ArnoldiWorkspaceMeta{eltype(temp),TV,TH,TVtmp,TQ}(V, H, V_tmp, Q)
end

function get_eigenvalues(
    U::Gaugefield,
    D::AbstractDiracOperator;
    nev::Int = 1,
    which::Symbol = :LM,
    tol::Real = sqrt(eps(Float64)),
    mindim::Int = min(max(10, nev), length(arnoldi.V)-1),
    maxdim::Int = min(max(20, 2nev), length(arnoldi.V)-1),
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
        maxdim=maxdim,
        restarts=restarts,
    )
    return partialeigen_meta(schur)
end

function get_eigenvalues(
    U::Gaugefield,
    D::AbstractDiracOperator,
    arnoldi::ArnoldiWorkspaceMeta;
    nev::Int = 1,
    which::Symbol = :LM,
    tol::Real = sqrt(eps(Float64)),
    mindim::Int = min(max(10, nev), length(arnoldi.V)-1),
    maxdim::Int = min(max(20, 2nev), length(arnoldi.V)-1),
    restarts::Int = 200,
    ddaggerd::Bool = false,
)
    D_in = ddaggerd ? DdaggerD(D(U)) : D(U)
    schur, _ = partialschur!(
        D_in,
        arnoldi;
        nev=nev,
        which=which,
        tol=tol,
        mindim=mindim,
        maxdim=maxdim,
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
    mindim::Int = min(max(10, nev), length(arnoldi.V)-1),
    maxdim::Int = min(max(20, 2nev), length(arnoldi.V)-1),
    restarts::Int = 200,
) where {T}
    n = LinearAlgebra.checksquare(D)
    @assert nev ≥ 1 "nev cannot be less than 1"
    @assert nev ≤ mindim ≤ maxdim < n "nev ≤ mindim ≤ maxdim < n must hold"
    _which = _symbol_to_target_meta(which)
    fill!(view(arnoldi.H, :, axes(arnoldi.H, 2)), zero(T)) 
    reinitialize_meta!(arnoldi, 0)
    _partialschur(D, arnoldi, mindim, maxdim, nev, tol, restarts, _which)
end

function reinitialize_meta!(arnoldi, j)
    V = arnoldi.V
    v = V[j+1]
    gaussian_pseudofermions!(v)
    rnorm = norm(v)

    if j==0
        mul!(v, 1 / rnorm)
        return true
    end

    η = sqrt(2) / 2
    Vprev = V[1:j]

    h = [dot(Vprev[i], v) for i in 1:j]
    for i in 1:j
        axpy!(-h[i], Vprev[i], v)
    end

    wnorm = norm(v)

    if wnorm < η * rnorm
        rnorm = wnorm
        for i in 1:j
            h[i] = dot(Vprev[i], v)
        end
        for i in 1:j
            axpy!(-h[i], Vprev[i], v)
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

function _partialschur(D, arnoldi, mindim, maxdim, nev, tol, restarts, which)
    H = arnoldi.H
    V = arnoldi.V
    V_tmp = arnoldi.V_tmp
    Q = arnoldi.Q

    T = eltype(H)
    x = zeros(T, maxdim)
    G = ArnoldiMethod.Reflector{T}(maxdim)
    ritz = ArnoldiMethod.RitzValues{T}(maxdim)
    isconverged = ArnoldiMethod.IsConverged(ritz, tol)
    ordering = ArnoldiMethod.get_order(which)
    groups = zeros(Int, maxdim)

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

    Vconverged = V[1:nconverged]
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

function orthogonalize_meta!(arnoldi, j)
    V, H = arnoldi.V, arnoldi.H

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
        correction = [dot(V[i], v) for i in 1:j]
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

get_order(which::SM) = ArnoldiMethod.OrderBy(abs)

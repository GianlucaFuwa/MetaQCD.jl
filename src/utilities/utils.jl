module Utils
    using LinearAlgebra
    using Random
    using StaticArrays

    export exp_iQ, exp_iQ_coeffs, exp_iQ_su3, B1, B2, Q
    export gen_SU3_matrix, is_SU3, is_su3
    export kenney_laub, proj_onto_SU3, make_submatrix, embed_into_SU3, tr
    export antihermitian, hermitian, traceless_antihermitian, traceless_hermitian
    export eye2, eye3, δ, ε_tensor, gaussian_su3_matrix
    export get_coords, move, SiteCoords

    const eye3 = @SArray [
        1.0+0.0im 0.0+0.0im 0.0+0.0im
        0.0+0.0im 1.0+0.0im 0.0+0.0im
        0.0+0.0im 0.0+0.0im 1.0+0.0im
    ]
    # Dirac-Basis γ matrices
    const γ1 = @SArray [
        0 0 0 1 
        0 0 1 0
        0 -1 0 0
        -1 0 0 0
    ]

    const γ2 = @SArray [
        0 0 0 -im 
        0 0 im 0
        0 im 0 0
        -im 0 0 0
    ]

    const γ3 = @SArray [
        0 0 1 0 
        0 0 0 -1
        -1 0 0 0
        0 1 0 0
    ]

    const γ4 = @SArray [
        1 0 0 0 
        0 1 0 0
        0 0 -1 0
        0 0 0 -1
    ]

    const eye2 = @SArray [
        1.0+0.0im 0.0+0.0im
        0.0+0.0im 1.0+0.0im
    ]

    const σ1 = @SArray [
        0 1
        1 0
    ]

    const σ2 = @SArray [
        0  -im
        im  0
    ]

    const σ3 = @SArray [
        1  0
        0 -1
    ]

    const σ_vec = [σ1, σ2, σ3]

    """
    Kronecker-Delta: \\
    δ(x, y) = \\
    {1, if x == y \\
    {0, else
    """
    function δ(x, y)
        return x == y
    end

    """
    Implementation of ε-tensor from: https://github.com/JuliaMath/Combinatorics.jl
    """
    function ε_tensor(p::NTuple{N, Int}) where {N}
		todo = Vector{Bool}(undef, N)
        todo .= true
		first = 1
		cycles = flips = 0
		
		while cycles + flips < N
			first = coalesce(findnext(todo, first), 0)
			(todo[first] = !todo[first]) && return 0
			j = p[first]
			cycles += 1
			
			while j != first
				(todo[j] = !todo[j]) && return 0
				j = p[j]
				flips += 1
			end
		end
		
		return iseven(flips) ? 1 : -1
	end

    function LinearAlgebra.tr(A::AbstractMatrix, B::AbstractMatrix)
        trace = 0.0 + 0.0im

        for i in 1:3
            @simd for j in 1:3
                @inbounds trace += A[i,j] * B[j,i]
            end
        end

        return trace
    end

    include("exp.jl")

    include("projections.jl")

    include("sitecoords.jl")

end
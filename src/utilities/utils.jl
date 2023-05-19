module Utils
    using LinearAlgebra
    using Random
    using StaticArrays

    import LinearAlgebra: tr

    export calc_coefficients, exp_iQ, gen_SU3_matrix, is_SU3, is_su3, gen_SU3_matrix
    export kenney_laub, proj_onto_SU3, make_submatrix, embed_into_SU3, tr
    export antihermitian, hermitian, traceless_antihermitian, traceless_hermitian
    export eye2, eye3, δ, ε_tensor
    export get_coords, move, SiteCoords, to_indices

    const eye3 = @SArray [
        1 0 0
        0 1 0
        0 0 1
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
        1 0
        0 1
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

    function LinearAlgebra.tr(A::SMatrix{3,3,ComplexF64,9}, B::SMatrix{3,3,ComplexF64,9})
        trace = 0.0

        for i in 1:3
            for j in 1:3
                trace += A[i,j] * B[j,i]
            end
        end

        return trace
    end

    include("exp.jl")

    include("projections.jl")

    include("sitecoords.jl")

end
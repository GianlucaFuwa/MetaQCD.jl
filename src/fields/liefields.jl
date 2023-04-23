module Liefields
    using Random
    using LinearAlgebra
    using StaticArrays
    using Base.Threads
    using Polyester

    using ..Utils
    import ..Gaugefields: Gaugefield

    struct Liefield
        P::Vector{Array{SMatrix{3,3,ComplexF64,9},4}}
		NX::Int64
		NY::Int64
		NZ::Int64
		NT::Int64
		NV::Int64
		NC::Int64	

		function Liefield(NX, NY, NZ, NT)
            NV = NX*NY*NZ*NT
			P = Vector{Array{SMatrix{3,3,ComplexF64,9},4}}(undef, 0)
			for μ = 1:4
                Pμ = Array{SMatrix{3,3,ComplexF64,9},4}(undef, NX, NY, NZ, NT)
                fill!(Pμ, SMatrix{3,3}(zeros(ComplexF64,3,3)))
				push!(P, Pμ)
			end
			return new(P, NX, NY, NZ, NT, NV, 3)
		end

        function Liefield(g::Gaugefield)
            return Liefield(g.NX, g.NY, g.NZ, g.NT)
        end
	end

    function Base.setindex!(p::Liefield, v, μ) 
        p.P[μ] = v
		return nothing
    end

    function Base.getindex(p::Liefield, μ)
        return p.P[μ]
    end

    function Base.size(p::Liefield)
        return p.NX, p.NY, p.NZ, p.NT
    end

    function clear_P!(p::Liefield)
		NX,NY,NZ,NT = size(p)
		for it = 1:NT
			for iz = 1:NZ
				for iy = 1:NY
					for ix = 1:NX
                        for μ = 1:4
                            @inbounds p[μ][ix,iy,iz,it] = zeros(3,3)
                        end
					end
				end
			end
		end
		return nothing
	end

    function gaussianP!(p::Liefield, rng::Xoshiro = Xoshiro())
        NX, NY, NZ, NT = size(p)
        sq3 = sqrt(3)
        @batch for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for μ = 1:4
                            h = SVector{8,Float64}(randn(rng, 8))
                            p[μ][ix,iy,iz,it] = 0.5im * [
                                h[3]+h[8]/sq3  h[1]-im*h[2]   h[4]-im*h[5]
                                h[1]+im*h[2]  -h[3]+h[8]/sq3  h[6]-im*h[7]
                                h[4]+im*h[5]   h[6]+im*h[7]  -2*h[8]/sq3 
                            ]
                        end
                    end
                end
            end
        end
		return nothing
	end

    function trP2(p::Liefield)
        NX, NY, NZ, NT = size(p)
        space = 8
        trP2 = zeros(nthreads()*space)
        @batch for it = 1:NT
            for iz = 1:NZ
                for iy = 1:NY
                    for ix = 1:NX
                        for μ = 1:4
                            trP2[threadid()*space] += real( trAB(p[μ][ix,iy,iz,it]) )
                        end
                    end
                end
            end
        end 
		return sum(trP2)
    end

end

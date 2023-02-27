module Liefields
    using Random
    using LinearAlgebra
    using StaticArrays
    using Base.Threads

    import ..Utils:exp_iQ

    mutable struct Liefield
        P::Array{SMatrix{3,3,ComplexF64,9},4}
		NX::Int64
		NY::Int64
		NZ::Int64
		NT::Int64
		NV::Int64
		NC::Int64	

		function Liefield(NX,NY,NZ,NT)
            NV = NX*NY*NZ*NT
            P = Array{SMatrix{3,3,ComplexF64,9},4}(undef,NX,NY,NZ,NT)
			return new(P,NX,NY,NZ,NT,NV,3)
		end
	end

    function Base.setindex!(p::Liefield,v,ix,iy,iz,it) 
        p.P[ix,iy,iz,it] = v
		return nothing
    end

	function Base.setindex!(p::Liefield,v,ix,iy,iz,it,i1,i2) 
        p.P[ix,iy,iz,it][i1,i2] = v
		return nothing
    end

	function Base.setindex!(p::Liefield,v,site::CartesianIndex{4}) 
        p.P[site] = v
		return nothing
    end

    function Base.getindex(p::Liefield,ix,iy,iz,it)
        return p.P[ix,iy,iz,it]
    end

	function Base.getindex(p::Liefield,ix,iy,iz,it,i1,i2)
        return p.P[ix,iy,iz,it][i1,i2]
    end

    function Base.getindex(p::Liefield,site::CartesianIndex{4})
        return p.P[site]
    end

    function gaussianP!(p::Liefield,rng::Xoshiro)
        NX = p.NX
		NY = p.NY
		NZ = p.NZ
		NT = p.NT
        sq3 = sqrt(3)
        h = zeros(8)
        for it=1:NT
        for iz=1:NZ
        for iy=1:NY
        for ix=1:NX
            randn!(rng,h)
            p.P[ix,iy,iz,it] = [
                h[3]+h[8]/sq3  h[1]-im*h[2]   h[4]-im*h[5]
                h[1]+im*h[2]  -h[3]+h[8]/sq3  h[6]-im*h[7]
                h[4]+im*h[5]   h[6]+im*h[7]  -2*h[8]/sq3 
            ]
        end
        end
        end
        end
		return nothing
	end

    function trP2(p::Liefield)
        NX = p.NX
		NY = p.NY
		NZ = p.NZ
		NT = p.NT
        space = 8
        trP2 = zeros(ComplexF64,nthreads()*space)
        @threads for it=1:NT
        for iz=1:NZ
        for iy=1:NY
        for ix=1:NX
            trP2[threadid()*space] += p[ix,iy,iz,it][1,1]^2 + p[ix,iy,iz,it][2,2]^2 + p[ix,iy,iz,it][3,3]^2 +
                    2*p[ix,iy,iz,it][1,2]*p[ix,iy,iz,it][2,1] + 2*p[ix,iy,iz,it][1,3]*p[ix,iy,iz,it][3,1] + 2*p[ix,iy,iz,it][2,3]*p[ix,iy,iz,it][3,2]
        end
        end
        end
        end 
		return real(sum(trP2))
    end

end

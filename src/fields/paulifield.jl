struct Paulifield{B,T,M,C,A} <: AbstractField{B,T,M,A} # So we can overload LinearAlgebra.det on the even-odd diagonal
    U::A
    csw::Float64
    NX::Int64 # Number of lattice sites in the x-direction
    NY::Int64 # Number of lattice sites in the y-direction
    NZ::Int64 # Number of lattice sites in the z-direction
    NT::Int64 # Number of lattice sites in the t-direction
    NV::Int64 # Total number of lattice sites
    NC::Int64 # Number of colors
    ND::Int64 # Number of dirac indices

    topology::FieldTopology # Info regarding MPI topology
    function Paulifield(
        f::TF, csw; inverse=false
    ) where {B,T,TF<:Union{Spinorfield{B,T},SpinorfieldEO{B,T}}}
        if f isa SpinorfieldEO
            numprocs_cart = f.parent.topology.numprocs_cart
            halo_width = f.parent.topology.halo_width
        else
            numprocs_cart = f.topology.numprocs_cart
            halo_width = f.topology.halo_width
        end

        NX, NY, NZ, NT = dims(f)
        _NT = inverse ? NT ÷ 2 : NT
        topology = FieldTopology(numprocs_cart, halo_width, (NX, NY, NZ, _NT))
        NC = num_colors(f)
        ND = num_dirac(f)
        NV = NX * NY * NZ * NT
        C = csw == 0 ? false : true
        U = KA.zeros(B(), PauliMatrix{6,36,T}, NX, NY, NZ, _NT)

        A = typeof(U)
        return new{B,T,false,C,A}(U, csw, NX, NY, NZ, NT, NV, NC, ND, topology)
    end
end

@inline global_dims(p::Paulifield) = NTuple{4,Int64}((p.NX, p.NY, p.NZ, p.NT))
num_colors(p::Paulifield) = p.NC

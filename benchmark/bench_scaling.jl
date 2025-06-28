using MetaQCD, MetaQCD.Utils, MPI, LinearAlgebra

numprocs_cart = (1, 1, 1, mpi_size())
halo_width = 1
U = Gaugefield{CPU,Float64,WilsonGaugeAction}(4, 4, 4, 4, 6.0; numprocs_cart, halo_width)
load_field!(BridgeFormat(), U, "../test/testconf_mpi")
ϕ = Spinorfield(U; staggered=true); ψ = Spinorfield(U; staggered=true);
D = StaggeredDiracOperator(U, 0.01);

mul!(ψ, (D(U)), ϕ)

sout = mpi_amroot() ? stdout : devnull
redirect_stdout(sout) do
    @benchmark mul!($ψ, $(D(U)), $ϕ)
end


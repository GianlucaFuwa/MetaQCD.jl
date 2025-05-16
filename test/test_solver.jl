using MetaQCD
using MetaQCD.Utils
using Random

function test_solver(
    backend=CPU;
    nprocs_cart=(1, 1, 1, 1),
    halo_width=1,
    dirac="staggered",
    mass=0.01,
    csw=1.78,
    eoprec=false,
    single_flavor=false,
)
    if mpi_amroot()
        println("Fermion derivative test [$dirac]")
    end

    Random.seed!(123 * (mpi_myrank() + 1))
    MetaQCD.MetaIO.set_global_logger!(4, nothing; tc=true)
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    U = Gaugefield{CPU,Float64,WilsonGaugeAction}(
        NX, NY, NZ, NT, 6.0, nprocs_cart, halo_width
    )
    random_gauges!(U)

    # filename = if nprocs_cart != (1, 1, 1, 1)
    #     pkgdir(MetaQCD, "test", "testconf_mpi")
    # else
    #     pkgdir(MetaQCD, "test", "testconf.txt")
    # end

    # load_config!(BridgeFormat(), U, filename)

    if backend !== CPU
        U = MetaQCD.to_backend(backend, U)
    end

    is_staggered = contains(dirac, "staggered")
    is_hoelbling = dirac ∈ ("staggered-h1234", "staggered-h1324", "staggered-h1342")

    ψ = if eoprec
        even_odd(Spinorfield(U; staggered=is_staggered))
    else
        Spinorfield(U; staggered=is_staggered)
    end

    spectral_bound, Nf = if is_staggered && !is_hoelbling
        (mass^2, 6.0), (single_flavor ? 1 : (eoprec ? 4 : 8))
    else
        (mass^2, 64.0), (single_flavor ? 1 : 2)
    end

    action = MetaQCD.DiracOperators.FermionAction(
        dirac*ifelse(eoprec, "_eo", ""),
        U,
        mass,
        Nf=Nf,
        bc_str="antiperiodic",
        rhmc_spectral_bound=spectral_bound,
        rhmc_order_action=15,
        cg_tol_action=1e-16,
        cg_maxiters_action=1000,
        wilson_csw=csw,
    )
    @show action

    sample_pseudofermions!(ψ, action, U)
    calc_fermion_action(action, U, ψ)
    return nothing
end

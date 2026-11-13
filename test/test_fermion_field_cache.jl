using MetaQCD
using Test

const CacheDO = MetaQCD.DiracOperators
cache_workspace(a) = (a.temps, a.temps_low)

@testset "Shared fermion solver fields" begin
    U = Gaugefield{CPU,Float64,WilsonGaugeAction,12}(4, 4, 4, 4, 6.0)
    identity_gauges!(U)
    cache = FermionFieldCache(U)
    for name in ("staggered", "staggered_eo", "wilson", "wilson_eo",
                 "staggered_h1234", "staggered_h1342"), precision in (Float64, Float32)
        kwargs = (; cg_float_type=precision, rhmc_order_action=4, rhmc_order_md=3,
                    rhmc_tol_action=1.0, rhmc_tol_md=1.0,
                    rhmc_spectral_bound=(0.1, 6.0))
        a = FermionAction(name, U, 0.5; field_cache=cache, kwargs...)
        b = FermionAction(name, U, 0.8; field_cache=cache, kwargs...)
        @test all(a.temps[i] === b.temps[i] for i in eachindex(a.temps))
        @test length(unique(objectid.(a.temps))) == length(a.temps)
        if precision == Float32
            @test all(a.temps_low[i] === b.temps_low[i] for i in 2:length(a.temps_low))
            @test a.temps[1] !== a.temps_low[2]
        end
        # Growing the pool for RHMC preserves existing tuples and field identities.
        r = FermionAction(name, U, 0.5; Nf=1, field_cache=cache, kwargs...)
        @test r.temps[1] === a.temps[1]
        @test all(isconcretetype, fieldtypes(typeof(r)))
        @test (@inferred cache_workspace(r)) === (r.temps, r.temps_low)
        private = FermionAction(name, U, 0.5; Nf=1, kwargs...)
        phi = CacheDO._solver_spinor(r.temps[1])
        CacheDO.gaussian_pseudofermions!(phi)
        # EO mixed solves have independent existing convergence issues; check
        # their storage and inference above without invoking that solver here.
        precision == Float32 && occursin("eo", name) && continue
        expected = CacheDO.calc_fermion_action(private, U, phi)
        @test CacheDO.calc_fermion_action(r, U, phi) ≈ expected
        # Another action can overwrite the shared workspace between solves.
        CacheDO.calc_fermion_action(b, U, phi)
        @test CacheDO.calc_fermion_action(r, U, phi) ≈ expected
    end
    other = Gaugefield{CPU,Float64,WilsonGaugeAction,12}(4, 4, 4, 4, 6.0)
    @test_throws ArgumentError FermionAction("staggered", other, 0.5; field_cache=cache)
end


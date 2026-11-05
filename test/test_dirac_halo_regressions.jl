pushfirst!(LOAD_PATH, dirname(@__DIR__))
import MetaQCD.DiracOperators as DO
import MetaQCD.Fields as F

using LinearAlgebra
using MetaQCD
using MetaQCD.Utils
using Random
using StaticArrays
using Test
const BTEST = if get(ENV, "METAQCD_TEST_BACKEND", "cpu") == "rocm"
    @eval using AMDGPU
    ROCBackend
else
    CPU
end
mpi_init()
const GRIDTEST = mpi_size() == 4 ? (1, 2, 2, 1) : (1, 1, 1, 1)
@test mpi_size() in (1, 4)
@inline halo_value(s, n) =
    mod1(s[1], n) + 10mod1(s[2], n) + 100mod1(s[3], n) + 1000mod1(s[4], n)

@testset "Face/corner validity, plans and fused Wilson ($BTEST)" begin
    u = Gaugefield{BTEST,Float64,WilsonGaugeAction,12}(
        8, 8, 8, 8, 6.0; numprocs_cart=GRIDTEST, halo_width=2
    )
    phi = Spinorfield(u; staggered=true)
    if mpi_size() > 1
        for offset in (0.0, 7.0)
            F.parallelfor(
                eachindex(phi), BTEST, Val(true), (), (phi,), (phi, offset)
            ) do site, (phi, offset)
                x = halo_value(site, 8) + offset
                phi[site] = SVector(ComplexF64(x), ComplexF64(2x), ComplexF64(-x))
            end
            @test !F.halo_is_valid(phi, Val(false))
            F.update_halo!((phi,); do_edges=Val(false))
            @test F.halo_is_valid(phi, Val(false))
            @test !F.halo_is_valid(phi, Val(true))
            err = F.parallelfor_sum(
                phi.topology.bulk_sites_padded,
                0.0,
                BTEST,
                Val(false),
                (),
                (),
                (phi, offset),
            ) do init, site, (phi, offset)
                outside = count(d -> !(site[d] in phi.topology.bulk_sites.indices[d]), 1:4)
                outside > 1 && return 0.0
                x = halo_value(site, 8) + offset
                return sum(
                    abs2,
                    phi[site] - SVector(ComplexF64(x), ComplexF64(2x), ComplexF64(-x)),
                )
            end
            @test mpi_allreduce(err, +, phi.topology.comm_cart) == 0
            plan = F.halo_plan(phi, Val(false))
            @test F.halo_plan(phi, Val(false)) === plan
            @test sum(b.count for b in plan.mpi_sendbufs) < sum(length, plan.sendbufs)
            F.update_halo!((phi,); do_edges=Val(true))
            @test F.halo_is_valid(phi, Val(true))
            err = F.parallelfor_sum(
                phi.topology.bulk_sites_padded,
                0.0,
                BTEST,
                Val(false),
                (),
                (),
                (phi, offset),
            ) do init, site, (phi, offset)
                x = halo_value(site, 8) + offset
                return sum(
                    abs2,
                    phi[site] - SVector(ComplexF64(x), ComplexF64(2x), ComplexF64(-x)),
                )
            end
            @test mpi_allreduce(err, +, phi.topology.comm_cart) == 0
        end
        subset = CartesianIndices((1:4, phi.topology.bulk_sites.indices[2:4]...))
        F.invalidate_halo!(phi)
        @test !F.can_overlap_halo(subset, (phi,), Val(false))
        subset_out = similar(phi)
        F.parallelfor(
            subset, BTEST, Val(true), (phi,), (subset_out,), (phi, subset_out)
        ) do site, (phi, out)
            out[site] = phi[site]
        end
        err = F.parallelfor_sum(
            eachindex(phi), 0.0, BTEST, Val(false), (), (), (phi, subset_out)
        ) do init, site, (phi, out)
            expected = site[1] <= 4 ? phi[site] : zero(phi[site])
            sum(abs2, out[site] - expected)
        end
        @test mpi_allreduce(err, +, phi.topology.comm_cart) == 0
        clone = deepcopy(phi)
        @test F.halo_plan(clone, Val(false)) !== F.halo_plan(phi, Val(false))
    end

    # Exercise all eight disjoint boundary slabs (as on a 4D process grid)
    # even though this test job has only four ranks.
    counter = similar(phi)
    borders = F.get_border_iterators(
        counter.topology.bulk_sites, 1, (true, true, true, true)
    )
    F._parallelfor(
        (site, captured) -> begin
            field = captured[1]
            field[site] = SVector(1.0 + 0im, 1.0 + 0im, 1.0 + 0im)
        end,
        (counter,),
        borders,
        BTEST,
        256,
    )
    err = F.parallelfor_sum(
        eachindex(counter), 0.0, BTEST, Val(false), (), (), (counter,)
    ) do init, site, (field,)
        boundary = any(
            d ->
                site[d] == first(field.topology.bulk_sites.indices[d]) ||
                    site[d] == last(field.topology.bulk_sites.indices[d]),
            1:4,
        )
        abs2(field[site][1] - (boundary ? 1 : 0))
    end
    @test err == 0
    @test F._parallelfor(identity, (counter,), (), BTEST, 0) === nothing

    Ucpu = Gaugefield{CPU,Float64,WilsonGaugeAction,12}(
        4, 4, 4, 4, 6.0; numprocs_cart=GRIDTEST, halo_width=2
    )
    load_field!(
        BridgeFormat(),
        Ucpu,
        pkgdir(MetaQCD, "test", mpi_size() == 1 ? "testconf.txt" : "testconf_mpi"),
    )
    Random.seed!(123 + mpi_myrank())
    pcpu = Spinorfield(Ucpu; halo_width=1)
    gaussian_pseudofermions!(pcpu)
    F.update_halo!((Ucpu, pcpu); do_edges=Val(true))
    U = BTEST == CPU ? Ucpu : convert_field(BTEST, Ucpu)
    p = BTEST == CPU ? pcpu : convert_field(BTEST, pcpu)
    F.invalidate_halo!(p)
    out = similar(p)
    for csw in (0.0, 1.78)
        dcpu = WilsonDiracOperator(Ucpu, 0.13; csw)(Ucpu)
        d = BTEST == CPU ? dcpu : WilsonDiracOperator(U, 0.13; csw)(U)
        for sign in (1, -1)
            # Reference is the original hopping + mass + six Clover additions.
            reference = map(collect(eachindex(pcpu))) do site
                value =
                    DO.wilson_kernel(
                        Ucpu, pcpu, site, dcpu.boundary_condition, Float64, Val(sign)
                    ) + 4.13pcpu[site]
                if csw != 0
                    for i in 1:6
                        value += DO.clover_kernel(
                            pcpu, dcpu.Fμν, site, Val(i), -csw / 2, Float64
                        )
                    end
                end
                value
            end
            for repeat in 1:3
                F.invalidate_halo!(p)
                mul!(out, sign == 1 ? d : adjoint(d), p)
            end
            F.device_synchronize(BTEST())
            actual = BTEST == CPU ? out : convert_field(CPU, out)
            maxerror = maximum(
                norm(actual[site] - v) for (site, v) in zip(eachindex(pcpu), reference)
            )
            @test mpi_allreduce(maxerror, max, Ucpu.topology.comm_cart) < 1e-10
        end
    end
end

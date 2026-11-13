# Run independently with: julia --project test/test_halo_gpu_control.jl
# Exercise GPU host-side control flow with real MPI and a CPU-backed test backend.
using MetaQCD
using Test

module HaloControlTests
using Test
import MetaQCD.Fields as F
import MetaQCD.Utils as U
const MPI = U.MPI
U.mpi_init()

struct TestBackend end
struct TestField{T} <: F.AbstractField{TestBackend,Float64,true}
    topology::T
    halo_valid::Base.RefValue{Bool}
    id::Int
    received::Dict{Int,Vector{Float64}}
end
F.default_stream(::TestBackend) = 0
F.get_stream(::TestBackend, i) = i
F.get_sendstream(::TestBackend, dir, dim, id) = 100id + 10dim + dir
F.get_readstream(::TestBackend, dir, dim, id) = 1000id + 10dim + dir
F.synchronize(::TestBackend, stream) = nothing
F.allocate_commstreams!(::TestBackend, fields) = nothing
F.array_type(::Type{TestBackend}) = Array
# Return a fresh allocation on EVERY lookup, as host-staged GPU receives do.
F.get_recv_buf(::TestField, i) = zeros(3)
function F.create_sendbuf!(u::TestField, sites, dim, dir; stream=0)
    @test stream != 0
    return fill(Float64(100u.id + 10dim + dir), 3)
end
function F.fill_halo!(u::TestField, buf, sites; stream=0)
    @test stream != 0
    u.received[sites] = copy(buf)
    return nothing
end
const launches = Ref(0)
function F.launch_foreachindex_global!(::TestBackend, f, captured, itrs, block, stream)
    launches[] += 1
    for itr in itrs, site in itr
        f(site, captured)
    end
end
function F.launch_foreachindex_reduce_global!(::TestBackend, init, op, f, captured, itrs, block, stream)
    launches[] += 1
    result = init
    for itr in itrs, site in itr
        result = op(result, f(init, site, captured))
    end
    return result
end

@testset "GPU halo host control (real MPI, CPU-backed backend)" begin
    comm = MPI.Cart_create(MPI.COMM_SELF, (1, 1, 1, 1); periodic=(true, true, true, true))
    bulk = CartesianIndices((3:10, 7:14, 2:7, 5:10))
    parts = (2, 2, 1, 1)
    borders = F.get_border_iterators(bulk, 1, parts .> 1)
    topo = (; comm_cart=comm, numprocs_cart=parts, halo_width=1,
        border_sites=ntuple(d -> (2d-1, 2d), 4),
        halo_sites=ntuple(d -> (2d-1, 2d), 4), border_iterators=borders)
    makefield(id) = TestField(topo, Ref(false), id, Dict{Int,Vector{Float64}}())

    for edges in (false, true)
        fields = (makefield(1), makefield(2))
        F.start_halo_update!(fields; do_edges=Val(edges))
        for u in fields
            @test F.halo_is_valid(u)
            @test length(u.received) == 4
            for dim in 1:2, dir in 1:2
                @test u.received[2dim-2+dir] == fill(100u.id + 10dim + 3-dir, 3)
            end
        end
        # Already-valid fields must not be packed or communicated again.
        empty!(fields[1].received)
        F.start_halo_update!(fields; do_edges=Val(edges))
        @test isempty(fields[1].received)
        # Mixed validity: filtering must preserve matching field tags and streams.
        F.invalidate_halo!(fields[2])
        F.start_halo_update!(fields; do_edges=Val(edges))
        @test isempty(fields[1].received)
        @test length(fields[2].received) == 4
    end

    for itr in (bulk, CartesianIndices((bulk.indices..., 1:4)))
        u = makefield(1)
        visited = Dict{eltype(itr),Int}()
        F.parallelfor(itr, TestBackend, Val(true), Val(true), (u,), (), (visited,)) do site, (v,)
            v[site] = get(v, site, 0) + 1
        end
        @test length(visited) == length(itr)
        @test all(==(1), values(visited))
        F.invalidate_halo!(u)
        launches[] = 0
        result = F.parallelfor_sum(itr, 0.0, TestBackend, Val(true), Val(true), (u,), (), ()) do init, site, captured
            1.0
        end
        @test result == length(itr)
        @test launches[] == 2 # one interior and one boundary reduction
        launches[] = 0
        result = F.parallelfor_sum(itr, 0.0, TestBackend, Val(true), Val(true), (u,), (), ()) do init, site, captured
            1.0
        end
        @test result == length(itr)
        @test launches[] == 1 # valid halos bypass splitting and communication
    end
    empty_itr = CartesianIndices((1:0, 1:2, 1:2, 1:2))
    @test F._parallelfor(identity, (), empty_itr, TestBackend, 0) === nothing
    @test F._parallelfor_sum(identity, (), empty_itr, 0.0, TestBackend, 0) == 0.0
    # Under mpiexec this also tests actual rank-to-rank transfers and tag pairing.
    nranks = MPI.Comm_size(MPI.COMM_WORLD)
    ring = MPI.Cart_create(MPI.COMM_WORLD, (1, 1, 1, nranks);
                           periodic=(true, true, true, true))
    rank = MPI.Comm_rank(ring)
    ring_topo = merge(topo, (; comm_cart=ring, numprocs_cart=(1, 1, 1, max(2, nranks))))
    for edges in (false, true)
        u = TestField(ring_topo, Ref(false), rank+1, Dict{Int,Vector{Float64}}())
        F.update_halo!((u,); do_edges=Val(edges))
        nbrs = MPI.Cart_shift(ring, 3, 1)
        for dir in 1:2
            @test u.received[6+dir] == fill(100(nbrs[dir]+1) + 40 + 3-dir, 3)
        end
    end
    MPI.free(ring)
    MPI.free(comm)
end
end

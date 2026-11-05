# Halo/Dirac optimization regression checks

The exchange implementation now caches its topology, MPI buffers and requests
per field and separately tracks valid faces and valid corners. A face exchange
packs all faces in one kernel and sends only their populated prefixes. It does
not certify corners. Full exchanges propagate extended faces by dimension.

The overlap launcher uses GPU events for producer/fill/result dependencies,
one shared compute queue, and one fused boundary launch. Sends are completed
while boundary work can execute. The receive-buffer reuse fence and the
pack-to-MPI readiness fence remain: ordinary GPU-aware MPI is not stream ordered.
Public `update_halo!` still completes on return; ordinary GPU kernels return
with results ordered on their caller's stream.

The normal and clover Wilson operators fuse hopping, mass and local clover
terms. Their spinor exchange only needs faces; gauge-dependent clover setup
still requests full corners. Benchmark setup now constructs `DU = D(U)` before
timing. The clover byte count excludes field-strength setup and describes the
fused application's traffic, so its new effective TB/s values should not be
compared directly with old clover files using the previous byte convention.
TFLOP/s and application times retain their existing work convention.

## Running the tests

Use an environment that can load MetaQCD and, for ROCm tests, AMDGPU. Launch
with the MPI binary selected by MPI.jl, not a potentially different system MPI.
For example, from the repository root:

```julia
using MetaQCD
cmd = `$(MetaQCD.Utils.MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project test/test_halo_operators_mpi.jl`
run(addenv(cmd, "METAQCD_TEST_BACKEND"=>"rocm", "JULIA_NUM_THREADS"=>"1"))
```

`METAQCD_TEST_BACKEND=cpu` selects CPU. The measurement fixture is 16⁴ and the
derivative fixture defaults to 16⁴. Set `METAQCD_DERIVATIVE_N=4` for the small
fixture. `METAQCD_TEST_PART` can select `measurements`, `derivatives`, or `all`.
The four-process grid is `(1,2,2,1)`, preserving the Polyakov-loop test's
unpartitioned time dimension.

`test_dirac_halo_regressions.jl` uses the same backend setting and four-process
launcher. It tests:

- face-only exchange, then a corner request without an intervening invalidation;
- invalidation after changing bulk data and reuse of cached communication plans;
- deep-copy isolation of exchange plans;
- safe fallback for subset iterators;
- fusion of eight disjoint boundary slabs and empty-iterator handling;
- normal/clover Wilson D and D† against the original hopping + mass + six-clover
  formulas, including repeated queued applications and unequal input halo widths.

The measurement test's standalone entry point no longer runs automatically when
included by another test file. The derivative fixture explicitly invalidates
perturbed fields and reduces link projections from the owning rank; the owner
need not be rank zero. The runner allows scalar GPU reads only for the finite-
difference test's small link readbacks, restoring the default guard afterwards.
Existing observable and derivative assertions are preserved. The runner also
checks finite errors and an absolute maximum relative error below 1e-3.

## Validation in this workspace

All final commands exited successfully with `MPI_HIDE_COMMUNICATION=true`,
`MPI_IS_GPUAWARE=true`, `TUNE_KERNELS=true`, and one Julia thread per rank.
User preferences were not changed.

| Backend | MPI ranks | Test | Result |
|---|---:|---|---|
| CPU | 4 | `test_meas.jl`, 16⁴ | Passed |
| CPU | 4 | `test_derivative.jl`, 16⁴ | Passed |
| CPU | 4 | Halo/fused-operator regressions | 25 checks passed per rank |
| ROCm | 4 | `test_meas.jl`, 16⁴ | Passed |
| ROCm | 4 | `test_derivative.jl`, 16⁴ | Passed |
| ROCm | 4 | Halo/fused-operator regressions | 25 checks passed per rank |

The combined CPU run took 1m29s and the combined ROCm run took 29m10s including
compilation. Earlier CPU checks also passed the small 4⁴ derivative fixture.
The maximum finite-difference relative error was 8.00e-4 on CPU and 5.33e-4
on ROCm. Both passed the additional 1e-3 maximum-error guard for all derivative
terms, in addition to the existing test assertions.

Local logs (ignored by git) are `_halo_cpu.log`, `_halo_rocm.log`,
`_halo_cpu_focused.log`, and `_halo_rocm_focused.log` in this directory.

ROCm was accessed outside the filesystem sandbox on the local Radeon 880M.
Four ranks shared this GPU (`FORCE_SINGLE_GPU=true`); these are correctness
checks, not Hunter multi-APU performance measurements. CUDA was not executed.

An initial GPU run completed measurements but hit the finite-difference test's
scalar-readback guard; its blocked MPI ranks were stopped. The final runner
correctly permits those small test readbacks and restores the guard afterwards.
Both requested suites then completed normally on the final implementation.

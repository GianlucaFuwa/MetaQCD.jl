# Benchmark result

* Pull request commit: [`c03f7727ba8e1c3f56da410728938ff65a34b5c8`](https://github.com/GianlucaFuwa/MetaQCD.jl/commit/c03f7727ba8e1c3f56da410728938ff65a34b5c8)
* Pull request: <https://github.com/GianlucaFuwa/MetaQCD.jl/pull/17> (Feature+Refactor: Distribute fields across processes with MPI)

# Judge result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmarks:
    - Target: 21 Sep 2024 - 13:47
    - Baseline: 21 Sep 2024 - 13:50
* Package commits:
    - Target: 12d7a9
    - Baseline: 20fd11
* Julia commits:
    - Target: 8e5136
    - Baseline: 8e5136
* Julia command flags:
    - Target: None
    - Baseline: None
* Environment variables:
    - Target: `OMP_NUM_THREADS => 1` `JULIA_NUM_THREADS => 2`
    - Baseline: `OMP_NUM_THREADS => 1` `JULIA_NUM_THREADS => 2`

## Results
A ratio greater than `1.0` denotes a possible regression (marked with :x:), while a ratio less
than `1.0` denotes a possible improvement (marked with :white_check_mark:). Only significant results - results
that indicate possible regressions or improvements - are shown below (thus, an empty table means that all
benchmark results remained invariant between builds).

| ID                                                                          | time ratio                   | memory ratio |
|-----------------------------------------------------------------------------|------------------------------|--------------|
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` | 0.91 (5%) :white_check_mark: |   1.00 (1%)  |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` | 0.92 (5%) :white_check_mark: |   1.00 (1%)  |
| `["meas", "measurements", "Float32, poly"]`                                 | 0.76 (5%) :white_check_mark: | Inf (1%) :x: |
| `["meas", "measurements", "Float64, poly"]`                                 | 0.87 (5%) :white_check_mark: | Inf (1%) :x: |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["dirac", "MetaQCD.DiracOperators.StaggeredDiracOperator"]`
- `["dirac", "MetaQCD.DiracOperators.WilsonDiracOperator"]`
- `["gauge", "1HB + 4OR"]`
- `["meas", "measurements"]`

## Julia versioninfo

### Target
```
Julia Version 1.9.4
Commit 8e5136fa297 (2023-11-14 08:46 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 22.04.5 LTS
  uname: Linux 6.8.0-1014-azure #16~22.04.1-Ubuntu SMP Thu Aug 15 21:31:41 UTC 2024 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3243 MHz       1901 s          0 s        119 s       4206 s          0 s
       #2  2445 MHz       1961 s          0 s        118 s       4159 s          0 s
       #3  2590 MHz       1708 s          0 s        138 s       4387 s          0 s
       #4  2590 MHz       2636 s          0 s        104 s       3507 s          0 s
  Memory: 15.615272521972656 GB (13511.25390625 MB free)
  Uptime: 627.23 sec
  Load Avg:  1.88  1.47  0.76
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

### Baseline
```
Julia Version 1.9.4
Commit 8e5136fa297 (2023-11-14 08:46 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 22.04.5 LTS
  uname: Linux 6.8.0-1014-azure #16~22.04.1-Ubuntu SMP Thu Aug 15 21:31:41 UTC 2024 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3040 MHz       2334 s          0 s        136 s       5299 s          0 s
       #2  3030 MHz       2502 s          0 s        135 s       5147 s          0 s
       #3  3239 MHz       2414 s          0 s        158 s       5205 s          0 s
       #4  2445 MHz       3172 s          0 s        120 s       4501 s          0 s
  Memory: 15.615272521972656 GB (13562.97265625 MB free)
  Uptime: 781.99 sec
  Load Avg:  1.59  1.45  0.86
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Target result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 21 Sep 2024 - 13:47
* Package commit: 12d7a9
* Julia commit: 8e5136
* Julia command flags: None
* Environment variables: `OMP_NUM_THREADS => 1` `JULIA_NUM_THREADS => 2`

## Results
Below is a table of this job's results, obtained by running the benchmarks.
The values listed in the `ID` column have the structure `[parent_group, child_group, ..., key]`, and can be used to
index into the BaseBenchmarks suite to retrieve the corresponding benchmarks.
The percentages accompanying time and memory values in the below table are noise tolerances. The "true"
time/memory value for a given benchmark is expected to fall within this percentage of the reported value.
An empty cell means that the value was zero.

| ID                                                                           | time            | GC time | memory         | allocations |
|------------------------------------------------------------------------------|----------------:|--------:|---------------:|------------:|
| `["dirac", "MetaQCD.DiracOperators.StaggeredDiracOperator", "Float32"]`      |   3.925 ms (5%) |         |                |             |
| `["dirac", "MetaQCD.DiracOperators.StaggeredDiracOperator", "Float64"]`      |   4.254 ms (5%) |         |                |             |
| `["dirac", "MetaQCD.DiracOperators.StaggeredEOPreDiracOperator", "Float32"]` |   7.781 ms (5%) |         |                |             |
| `["dirac", "MetaQCD.DiracOperators.StaggeredEOPreDiracOperator", "Float64"]` |   8.456 ms (5%) |         |                |             |
| `["dirac", "MetaQCD.DiracOperators.WilsonDiracOperator", "Float32"]`         |  88.161 ms (5%) |         |                |             |
| `["dirac", "MetaQCD.DiracOperators.WilsonDiracOperator", "Float64"]`         |  92.363 ms (5%) |         |                |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]`  |    1.332 s (5%) |         |                |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]`  |    1.551 s (5%) |         |                |             |
| `["meas", "measurements", "Float32, ed"]`                                    | 384.230 ms (5%) |         |                |             |
| `["meas", "measurements", "Float32, gaction"]`                               | 136.771 ms (5%) |         |                |             |
| `["meas", "measurements", "Float32, plaq"]`                                  |  10.434 ms (5%) |         |                |             |
| `["meas", "measurements", "Float32, poly"]`                                  | 430.233 μs (5%) |         | 816 bytes (1%) |          14 |
| `["meas", "measurements", "Float32, topo"]`                                  | 437.439 ms (5%) |         |                |             |
| `["meas", "measurements", "Float32, wilson"]`                                | 582.361 ms (5%) |         |                |             |
| `["meas", "measurements", "Float64, ed"]`                                    | 394.832 ms (5%) |         |                |             |
| `["meas", "measurements", "Float64, gaction"]`                               | 149.218 ms (5%) |         |                |             |
| `["meas", "measurements", "Float64, plaq"]`                                  |  10.136 ms (5%) |         |                |             |
| `["meas", "measurements", "Float64, poly"]`                                  | 509.812 μs (5%) |         | 816 bytes (1%) |          14 |
| `["meas", "measurements", "Float64, topo"]`                                  | 414.272 ms (5%) |         |                |             |
| `["meas", "measurements", "Float64, wilson"]`                                | 682.656 ms (5%) |         |                |             |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["dirac", "MetaQCD.DiracOperators.StaggeredDiracOperator"]`
- `["dirac", "MetaQCD.DiracOperators.StaggeredEOPreDiracOperator"]`
- `["dirac", "MetaQCD.DiracOperators.WilsonDiracOperator"]`
- `["gauge", "1HB + 4OR"]`
- `["meas", "measurements"]`

## Julia versioninfo
```
Julia Version 1.9.4
Commit 8e5136fa297 (2023-11-14 08:46 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 22.04.5 LTS
  uname: Linux 6.8.0-1014-azure #16~22.04.1-Ubuntu SMP Thu Aug 15 21:31:41 UTC 2024 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3243 MHz       1901 s          0 s        119 s       4206 s          0 s
       #2  2445 MHz       1961 s          0 s        118 s       4159 s          0 s
       #3  2590 MHz       1708 s          0 s        138 s       4387 s          0 s
       #4  2590 MHz       2636 s          0 s        104 s       3507 s          0 s
  Memory: 15.615272521972656 GB (13511.25390625 MB free)
  Uptime: 627.23 sec
  Load Avg:  1.88  1.47  0.76
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Baseline result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 21 Sep 2024 - 13:50
* Package commit: 20fd11
* Julia commit: 8e5136
* Julia command flags: None
* Environment variables: `OMP_NUM_THREADS => 1` `JULIA_NUM_THREADS => 2`

## Results
Below is a table of this job's results, obtained by running the benchmarks.
The values listed in the `ID` column have the structure `[parent_group, child_group, ..., key]`, and can be used to
index into the BaseBenchmarks suite to retrieve the corresponding benchmarks.
The percentages accompanying time and memory values in the below table are noise tolerances. The "true"
time/memory value for a given benchmark is expected to fall within this percentage of the reported value.
An empty cell means that the value was zero.

| ID                                                                          | time            | GC time | memory | allocations |
|-----------------------------------------------------------------------------|----------------:|--------:|-------:|------------:|
| `["dirac", "MetaQCD.DiracOperators.StaggeredDiracOperator", "Float32"]`     |   3.925 ms (5%) |         |        |             |
| `["dirac", "MetaQCD.DiracOperators.StaggeredDiracOperator", "Float64"]`     |   4.329 ms (5%) |         |        |             |
| `["dirac", "MetaQCD.DiracOperators.WilsonDiracOperator", "Float32"]`        |  90.223 ms (5%) |         |        |             |
| `["dirac", "MetaQCD.DiracOperators.WilsonDiracOperator", "Float64"]`        |  95.181 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.457 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.677 s (5%) |         |        |             |
| `["meas", "measurements", "Float32, ed"]`                                   | 397.191 ms (5%) |         |        |             |
| `["meas", "measurements", "Float32, gaction"]`                              | 136.888 ms (5%) |         |        |             |
| `["meas", "measurements", "Float32, plaq"]`                                 |  10.424 ms (5%) |         |        |             |
| `["meas", "measurements", "Float32, poly"]`                                 | 563.812 μs (5%) |         |        |             |
| `["meas", "measurements", "Float32, topo"]`                                 | 421.386 ms (5%) |         |        |             |
| `["meas", "measurements", "Float32, wilson"]`                               | 583.025 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, ed"]`                                   | 396.414 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, gaction"]`                              | 150.249 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, plaq"]`                                 |  10.104 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, poly"]`                                 | 586.665 μs (5%) |         |        |             |
| `["meas", "measurements", "Float64, topo"]`                                 | 420.721 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, wilson"]`                               | 698.687 ms (5%) |         |        |             |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["dirac", "MetaQCD.DiracOperators.StaggeredDiracOperator"]`
- `["dirac", "MetaQCD.DiracOperators.WilsonDiracOperator"]`
- `["gauge", "1HB + 4OR"]`
- `["meas", "measurements"]`

## Julia versioninfo
```
Julia Version 1.9.4
Commit 8e5136fa297 (2023-11-14 08:46 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 22.04.5 LTS
  uname: Linux 6.8.0-1014-azure #16~22.04.1-Ubuntu SMP Thu Aug 15 21:31:41 UTC 2024 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3040 MHz       2334 s          0 s        136 s       5299 s          0 s
       #2  3030 MHz       2502 s          0 s        135 s       5147 s          0 s
       #3  3239 MHz       2414 s          0 s        158 s       5205 s          0 s
       #4  2445 MHz       3172 s          0 s        120 s       4501 s          0 s
  Memory: 15.615272521972656 GB (13562.97265625 MB free)
  Uptime: 781.99 sec
  Load Avg:  1.59  1.45  0.86
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Runtime information
| Runtime Info | |
|:--|:--|
| BLAS #threads | 2 |
| `BLAS.vendor()` | `lbt` |
| `Sys.CPU_THREADS` | 4 |

`lscpu` output:

    Architecture:                         x86_64
    CPU op-mode(s):                       32-bit, 64-bit
    Address sizes:                        48 bits physical, 48 bits virtual
    Byte Order:                           Little Endian
    CPU(s):                               4
    On-line CPU(s) list:                  0-3
    Vendor ID:                            AuthenticAMD
    Model name:                           AMD EPYC 7763 64-Core Processor
    CPU family:                           25
    Model:                                1
    Thread(s) per core:                   2
    Core(s) per socket:                   2
    Socket(s):                            1
    Stepping:                             1
    BogoMIPS:                             4890.86
    Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves user_shstk clzero xsaveerptr rdpru arat npt nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload umip vaes vpclmulqdq rdpid fsrm
    Virtualization:                       AMD-V
    Hypervisor vendor:                    Microsoft
    Virtualization type:                  full
    L1d cache:                            64 KiB (2 instances)
    L1i cache:                            64 KiB (2 instances)
    L2 cache:                             1 MiB (2 instances)
    L3 cache:                             32 MiB (1 instance)
    NUMA node(s):                         1
    NUMA node0 CPU(s):                    0-3
    Vulnerability Gather data sampling:   Not affected
    Vulnerability Itlb multihit:          Not affected
    Vulnerability L1tf:                   Not affected
    Vulnerability Mds:                    Not affected
    Vulnerability Meltdown:               Not affected
    Vulnerability Mmio stale data:        Not affected
    Vulnerability Reg file data sampling: Not affected
    Vulnerability Retbleed:               Not affected
    Vulnerability Spec rstack overflow:   Vulnerable: Safe RET, no microcode
    Vulnerability Spec store bypass:      Vulnerable
    Vulnerability Spectre v1:             Mitigation; usercopy/swapgs barriers and __user pointer sanitization
    Vulnerability Spectre v2:             Mitigation; Retpolines; STIBP disabled; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
    Vulnerability Srbds:                  Not affected
    Vulnerability Tsx async abort:        Not affected
    

| Cpu Property       | Value                                                      |
|:------------------ |:---------------------------------------------------------- |
| Brand              | AMD EPYC 7763 64-Core Processor                            |
| Vendor             | :AMD                                                       |
| Architecture       | :Unknown                                                   |
| Model              | Family: 0xaf, Model: 0x01, Stepping: 0x01, Type: 0x00      |
| Cores              | 16 physical cores, 16 logical cores (on executing CPU)     |
|                    | No Hyperthreading hardware capability detected             |
| Clock Frequencies  | Not supported by CPU                                       |
| Data Cache         | Level 1:3 : (32, 512, 32768) kbytes                        |
|                    | 64 byte cache line size                                    |
| Address Size       | 48 bits virtual, 48 bits physical                          |
| SIMD               | 256 bit = 32 byte max. SIMD vector size                    |
| Time Stamp Counter | TSC is accessible via `rdtsc`                              |
|                    | TSC runs at constant rate (invariant from clock frequency) |
| Perf. Monitoring   | Performance Monitoring Counters (PMC) are not supported    |
| Hypervisor         | Yes, Microsoft                                             |


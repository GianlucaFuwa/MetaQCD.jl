# Benchmark result

* Pull request commit: [`29b5e9f868e1cc5c74cf944ed92fe82c623bc379`](https://github.com/GianlucaFuwa/MetaQCD.jl/commit/29b5e9f868e1cc5c74cf944ed92fe82c623bc379)
* Pull request: <https://github.com/GianlucaFuwa/MetaQCD.jl/pull/17> (Feature+Refactor: Distribute fields across processes with MPI)

# Judge result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmarks:
    - Target: 13 Oct 2024 - 01:52
    - Baseline: 13 Oct 2024 - 01:54
* Package commits:
    - Target: 72632b
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

| ID                                                                          | time ratio | memory ratio |
|-----------------------------------------------------------------------------|------------|--------------|

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["gauge", "1HB + 4OR"]`

## Julia versioninfo

### Target
```
Julia Version 1.9.4
Commit 8e5136fa297 (2023-11-14 08:46 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 24.04.1 LTS
  uname: Linux 6.8.0-1015-azure #17-Ubuntu SMP Mon Sep  2 14:54:06 UTC 2024 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3236 MHz       1769 s          0 s        130 s       3550 s          0 s
       #2  3241 MHz       2392 s          0 s         99 s       2984 s          0 s
       #3  3237 MHz       2224 s          0 s        120 s       3164 s          0 s
       #4  3301 MHz       1833 s          0 s        117 s       3568 s          0 s
  Memory: 15.615283966064453 GB (13571.69140625 MB free)
  Uptime: 557.18 sec
  Load Avg:  1.85  1.51  0.79
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
      Ubuntu 24.04.1 LTS
  uname: Linux 6.8.0-1015-azure #17-Ubuntu SMP Mon Sep  2 14:54:06 UTC 2024 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3240 MHz       2331 s          0 s        145 s       4480 s          0 s
       #2  3245 MHz       2882 s          0 s        119 s       3980 s          0 s
       #3  3215 MHz       2848 s          0 s        135 s       4032 s          0 s
       #4  3241 MHz       2337 s          0 s        138 s       4550 s          0 s
  Memory: 15.615283966064453 GB (13586.4609375 MB free)
  Uptime: 708.08 sec
  Load Avg:  1.58  1.49  0.89
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Target result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 13 Oct 2024 - 1:52
* Package commit: 72632b
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float32"]`               |   7.668 ms (5%) |         |        |             |
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               |   8.687 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float32"]`                                         |   3.862 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float64"]`                                         |   4.178 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float32"]`                                            |  87.481 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float64"]`                                            |  96.085 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.279 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.647 s (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  10.381 ms (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  10.088 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 383.337 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 386.993 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 136.304 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 146.880 ms (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 415.618 μs (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 419.896 μs (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 404.321 ms (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 393.731 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 561.745 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 641.234 ms (5%) |         |        |             |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["dirac", "Staggered (Even-Odd preconditioned)"]`
- `["dirac", "Staggered"]`
- `["dirac", "Wilson"]`
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
      Ubuntu 24.04.1 LTS
  uname: Linux 6.8.0-1015-azure #17-Ubuntu SMP Mon Sep  2 14:54:06 UTC 2024 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3236 MHz       1769 s          0 s        130 s       3550 s          0 s
       #2  3241 MHz       2392 s          0 s         99 s       2984 s          0 s
       #3  3237 MHz       2224 s          0 s        120 s       3164 s          0 s
       #4  3301 MHz       1833 s          0 s        117 s       3568 s          0 s
  Memory: 15.615283966064453 GB (13571.69140625 MB free)
  Uptime: 557.18 sec
  Load Avg:  1.85  1.51  0.79
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Baseline result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 13 Oct 2024 - 1:54
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
| `["dirac", "MetaQCD.DiracOperators.StaggeredDiracOperator", "Float32"]`     |   3.888 ms (5%) |         |        |             |
| `["dirac", "MetaQCD.DiracOperators.StaggeredDiracOperator", "Float64"]`     |   4.266 ms (5%) |         |        |             |
| `["dirac", "MetaQCD.DiracOperators.WilsonDiracOperator", "Float32"]`        |  88.956 ms (5%) |         |        |             |
| `["dirac", "MetaQCD.DiracOperators.WilsonDiracOperator", "Float64"]`        |  92.664 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.303 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.593 s (5%) |         |        |             |
| `["meas", "measurements", "Float32, ed"]`                                   | 385.998 ms (5%) |         |        |             |
| `["meas", "measurements", "Float32, gaction"]`                              | 136.491 ms (5%) |         |        |             |
| `["meas", "measurements", "Float32, plaq"]`                                 |  10.402 ms (5%) |         |        |             |
| `["meas", "measurements", "Float32, poly"]`                                 | 555.871 μs (5%) |         |        |             |
| `["meas", "measurements", "Float32, topo"]`                                 | 406.704 ms (5%) |         |        |             |
| `["meas", "measurements", "Float32, wilson"]`                               | 552.619 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, ed"]`                                   | 384.096 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, gaction"]`                              | 147.728 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, plaq"]`                                 |  10.164 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, poly"]`                                 | 632.615 μs (5%) |         |        |             |
| `["meas", "measurements", "Float64, topo"]`                                 | 406.467 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, wilson"]`                               | 646.545 ms (5%) |         |        |             |

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
      Ubuntu 24.04.1 LTS
  uname: Linux 6.8.0-1015-azure #17-Ubuntu SMP Mon Sep  2 14:54:06 UTC 2024 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3240 MHz       2331 s          0 s        145 s       4480 s          0 s
       #2  3245 MHz       2882 s          0 s        119 s       3980 s          0 s
       #3  3215 MHz       2848 s          0 s        135 s       4032 s          0 s
       #4  3241 MHz       2337 s          0 s        138 s       4550 s          0 s
  Memory: 15.615283966064453 GB (13586.4609375 MB free)
  Uptime: 708.08 sec
  Load Avg:  1.58  1.49  0.89
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


# Benchmark result

* Pull request commit: [`9e43ee54349f08d0c5ad0c9a7574450e798c0a3d`](https://github.com/GianlucaFuwa/MetaQCD.jl/commit/9e43ee54349f08d0c5ad0c9a7574450e798c0a3d)
* Pull request: <https://github.com/GianlucaFuwa/MetaQCD.jl/pull/20> (Multiple timescale integrator)

# Judge result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmarks:
    - Target: 26 Jun 2025 - 17:50
    - Baseline: 26 Jun 2025 - 17:54
* Package commits:
    - Target: 07b78e8
    - Baseline: 8a6ac11
* Julia commits:
    - Target: 8e5136f
    - Baseline: 8e5136f
* Julia command flags:
    - Target: None
    - Baseline: None
* Environment variables:
    - Target: `OMP_NUM_THREADS => 1` `JULIA_NUM_THREADS => 2`
    - Baseline: `OMP_NUM_THREADS => 1` `JULIA_NUM_THREADS => 2`

## Results
A ratio greater than `1.0` denotes a possible regression (marked with :x:), while a ratio less
than `1.0` denotes a possible improvement (marked with :white_check_mark:). Brackets display [tolerances](https://juliaci.github.io/BenchmarkTools.jl/stable/manual/#Benchmark-Parameters) for the benchmark estimates. Only significant results - results
that indicate possible regressions or improvements - are shown below (thus, an empty table means that all
benchmark results remained invariant between builds).

| ID                                                                          | time ratio                   | memory ratio |
|-----------------------------------------------------------------------------|------------------------------|--------------|
| `["dirac", "Staggered", "Float32"]`                                         |               86.99 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Staggered", "Float64"]`                                         |               20.98 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Wilson", "Float32"]`                                            |                4.52 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Wilson", "Float64"]`                                            |                2.35 (5%) :x: | Inf (1%) :x: |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` | 0.93 (5%) :white_check_mark: | Inf (1%) :x: |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |                   0.96 (5%)  | Inf (1%) :x: |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |                1.43 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |                1.48 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Energy Density, Float32"]`                       |                1.87 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Energy Density, Float64"]`                       |                1.95 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    |                1.99 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    |                1.89 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        |                   1.00 (5%)  | Inf (1%) :x: |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        |                   1.02 (5%)  | Inf (1%) :x: |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      |                1.12 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      |                1.26 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             |                1.57 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             |                1.54 (5%) :x: | Inf (1%) :x: |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["dirac", "Staggered"]`
- `["dirac", "Wilson"]`
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
      Ubuntu 24.04.2 LTS
  uname: Linux 6.11.0-1015-azure #15~24.04.1-Ubuntu SMP Thu May  1 02:52:08 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3241 MHz       1852 s          0 s         91 s       9009 s          0 s
       #2  3169 MHz       2366 s          0 s         97 s       8496 s          0 s
       #3  3243 MHz       2189 s          0 s        156 s       8611 s          0 s
       #4  3231 MHz       3144 s          0 s        185 s       7627 s          0 s
  Memory: 15.620769500732422 GB (9183.12890625 MB free)
  Uptime: 1098.03 sec
  Load Avg:  1.84  1.54  0.85
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
      Ubuntu 24.04.2 LTS
  uname: Linux 6.11.0-1015-azure #15~24.04.1-Ubuntu SMP Thu May  1 02:52:08 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3242 MHz       2416 s          0 s        100 s      10609 s          0 s
       #2  3244 MHz       3009 s          0 s        102 s      10021 s          0 s
       #3  3244 MHz       3006 s          0 s        164 s       9962 s          0 s
       #4  3249 MHz       4384 s          0 s        197 s       8551 s          0 s
  Memory: 15.620769500732422 GB (13677.21875 MB free)
  Uptime: 1315.75 sec
  Load Avg:  1.69  1.56  1.0
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Target result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 26 Jun 2025 - 17:50
* Package commit: 07b78e8
* Julia commit: 8e5136f
* Julia command flags: None
* Environment variables: `OMP_NUM_THREADS => 1` `JULIA_NUM_THREADS => 2`

## Results
Below is a table of this job's results, obtained by running the benchmarks.
The values listed in the `ID` column have the structure `[parent_group, child_group, ..., key]`, and can be used to
index into the BaseBenchmarks suite to retrieve the corresponding benchmarks.
The percentages accompanying time and memory values in the below table are noise tolerances. The "true"
time/memory value for a given benchmark is expected to fall within this percentage of the reported value.
An empty cell means that the value was zero.

| ID                                                                          | time            | GC time | memory          | allocations |
|-----------------------------------------------------------------------------|----------------:|--------:|----------------:|------------:|
| `["dirac", "Staggered", "Float32"]`                                         | 333.809 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Staggered", "Float64"]`                                         |  88.012 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Wilson", "Float32"]`                                            | 396.418 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Wilson", "Float64"]`                                            | 213.327 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Wilson-Clover", "Float32"]`                                     |    2.098 s (5%) |         |   8.50 KiB (1%) |           2 |
| `["dirac", "Wilson-Clover", "Float64"]`                                     |    1.796 s (5%) |         |   8.50 KiB (1%) |           2 |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.261 s (5%) |         | 127.50 KiB (1%) |          80 |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.523 s (5%) |         | 127.50 KiB (1%) |          80 |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  14.832 ms (5%) |         |   1.45 KiB (1%) |           1 |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  14.951 ms (5%) |         |   1.45 KiB (1%) |           1 |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 718.015 ms (5%) |         |   6.23 KiB (1%) |           4 |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 728.137 ms (5%) |         |   6.23 KiB (1%) |           4 |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 272.083 ms (5%) |         |  10.17 KiB (1%) |           7 |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 273.534 ms (5%) |         |  10.17 KiB (1%) |           7 |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 422.107 μs (5%) |         |   1.59 KiB (1%) |           1 |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 545.607 μs (5%) |         |   1.59 KiB (1%) |           1 |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 452.940 ms (5%) |         |   4.50 KiB (1%) |           3 |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 479.976 ms (5%) |         |   4.50 KiB (1%) |           3 |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 877.083 ms (5%) |         |  25.50 KiB (1%) |          16 |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 988.855 ms (5%) |         |  25.50 KiB (1%) |          16 |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["dirac", "Staggered"]`
- `["dirac", "Wilson"]`
- `["dirac", "Wilson-Clover"]`
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
      Ubuntu 24.04.2 LTS
  uname: Linux 6.11.0-1015-azure #15~24.04.1-Ubuntu SMP Thu May  1 02:52:08 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3241 MHz       1852 s          0 s         91 s       9009 s          0 s
       #2  3169 MHz       2366 s          0 s         97 s       8496 s          0 s
       #3  3243 MHz       2189 s          0 s        156 s       8611 s          0 s
       #4  3231 MHz       3144 s          0 s        185 s       7627 s          0 s
  Memory: 15.620769500732422 GB (9183.12890625 MB free)
  Uptime: 1098.03 sec
  Load Avg:  1.84  1.54  0.85
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Baseline result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 26 Jun 2025 - 17:54
* Package commit: 8a6ac11
* Julia commit: 8e5136f
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float32"]`               |   7.706 ms (5%) |         |        |             |
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               |   8.825 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float32"]`                                         |   3.837 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float64"]`                                         |   4.196 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float32"]`                                            |  87.780 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float64"]`                                            |  90.851 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.351 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.594 s (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  10.376 ms (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  10.119 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 383.285 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 373.897 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 136.523 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 144.549 ms (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 420.714 μs (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 535.047 μs (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 405.324 ms (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 382.075 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 560.374 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 643.642 ms (5%) |         |        |             |

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
      Ubuntu 24.04.2 LTS
  uname: Linux 6.11.0-1015-azure #15~24.04.1-Ubuntu SMP Thu May  1 02:52:08 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3242 MHz       2416 s          0 s        100 s      10609 s          0 s
       #2  3244 MHz       3009 s          0 s        102 s      10021 s          0 s
       #3  3244 MHz       3006 s          0 s        164 s       9962 s          0 s
       #4  3249 MHz       4384 s          0 s        197 s       8551 s          0 s
  Memory: 15.620769500732422 GB (13677.21875 MB free)
  Uptime: 1315.75 sec
  Load Avg:  1.69  1.56  1.0
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
    BogoMIPS:                             4890.87
    Flags:                                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves user_shstk clzero xsaveerptr rdpru arat npt nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload umip vaes vpclmulqdq rdpid fsrm
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


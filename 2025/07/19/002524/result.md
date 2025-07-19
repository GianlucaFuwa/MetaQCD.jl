# Benchmark result

* Pull request commit: [`3fcd9ca931998d6685465a142256a4f5c8ed42f8`](https://github.com/GianlucaFuwa/MetaQCD.jl/commit/3fcd9ca931998d6685465a142256a4f5c8ed42f8)
* Pull request: <https://github.com/GianlucaFuwa/MetaQCD.jl/pull/20> (Multiple timescale integrator)

# Judge result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmarks:
    - Target: 19 Jul 2025 - 00:21
    - Baseline: 19 Jul 2025 - 00:25
* Package commits:
    - Target: 4264e93
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
| `["dirac", "Staggered", "Float32"]`                                         |               23.93 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Staggered", "Float64"]`                                         |               16.58 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Wilson", "Float32"]`                                            |                1.20 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Wilson", "Float64"]`                                            |                2.32 (5%) :x: | Inf (1%) :x: |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |                   0.96 (5%)  | Inf (1%) :x: |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` | 0.95 (5%) :white_check_mark: | Inf (1%) :x: |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |                1.11 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |                1.16 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Energy Density, Float32"]`                       |                1.23 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Energy Density, Float64"]`                       |                1.29 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    |                1.10 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    |                   1.03 (5%)  | Inf (1%) :x: |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        |                   1.03 (5%)  | Inf (1%) :x: |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        |                1.37 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      |                1.18 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      |                1.33 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             |                1.17 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             |                1.15 (5%) :x: | Inf (1%) :x: |

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
  uname: Linux 6.11.0-1018-azure #18~24.04.1-Ubuntu SMP Sat Jun 28 04:46:03 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3243 MHz       2257 s          0 s         94 s       3024 s          0 s
       #2  3241 MHz       1970 s          0 s         95 s       3312 s          0 s
       #3  3255 MHz       2090 s          0 s        103 s       3189 s          0 s
       #4  3244 MHz       3183 s          0 s         96 s       2095 s          0 s
  Memory: 15.620681762695312 GB (9099.25390625 MB free)
  Uptime: 540.07 sec
  Load Avg:  1.9  1.61  0.88
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
  uname: Linux 6.11.0-1018-azure #18~24.04.1-Ubuntu SMP Sat Jun 28 04:46:03 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3224 MHz       2668 s          0 s        104 s       4774 s          0 s
       #2  3242 MHz       2750 s          0 s         99 s       4701 s          0 s
       #3  3242 MHz       2870 s          0 s        114 s       4570 s          0 s
       #4  3249 MHz       4460 s          0 s        109 s       2978 s          0 s
  Memory: 15.620681762695312 GB (13689.79296875 MB free)
  Uptime: 757.49 sec
  Load Avg:  1.68  1.59  1.03
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Target result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 19 Jul 2025 - 00:21
* Package commit: 4264e93
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
| `["dirac", "Staggered", "Float32"]`                                         |  92.679 ms (5%) |         |  13.08 KiB (1%) |          12 |
| `["dirac", "Staggered", "Float64"]`                                         |  68.765 ms (5%) |         |  13.08 KiB (1%) |          12 |
| `["dirac", "Wilson", "Float32"]`                                            | 104.944 ms (5%) |         |  13.08 KiB (1%) |          12 |
| `["dirac", "Wilson", "Float64"]`                                            | 209.755 ms (5%) |         |  13.08 KiB (1%) |          12 |
| `["dirac", "Wilson-Clover", "Float32"]`                                     |    1.771 s (5%) |         |  26.16 KiB (1%) |          24 |
| `["dirac", "Wilson-Clover", "Float64"]`                                     |    1.969 s (5%) |         |  26.16 KiB (1%) |          24 |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.291 s (5%) |         | 390.75 KiB (1%) |        1408 |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.549 s (5%) |         | 390.75 KiB (1%) |        1408 |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  11.549 ms (5%) |         |   4.84 KiB (1%) |          17 |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  11.696 ms (5%) |         |   4.84 KiB (1%) |          17 |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 480.844 ms (5%) |         |  19.47 KiB (1%) |          74 |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 495.853 ms (5%) |         |  19.56 KiB (1%) |          74 |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 150.297 ms (5%) |         |  33.91 KiB (1%) |         119 |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 151.842 ms (5%) |         |  33.91 KiB (1%) |         119 |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 429.262 μs (5%) |         |   4.98 KiB (1%) |          19 |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 568.203 μs (5%) |         |   4.98 KiB (1%) |          19 |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 486.941 ms (5%) |         |  14.56 KiB (1%) |          53 |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 523.853 ms (5%) |         |  14.59 KiB (1%) |          53 |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 663.348 ms (5%) |         |  78.50 KiB (1%) |         304 |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 752.472 ms (5%) |         |  78.50 KiB (1%) |         304 |

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
  uname: Linux 6.11.0-1018-azure #18~24.04.1-Ubuntu SMP Sat Jun 28 04:46:03 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3243 MHz       2257 s          0 s         94 s       3024 s          0 s
       #2  3241 MHz       1970 s          0 s         95 s       3312 s          0 s
       #3  3255 MHz       2090 s          0 s        103 s       3189 s          0 s
       #4  3244 MHz       3183 s          0 s         96 s       2095 s          0 s
  Memory: 15.620681762695312 GB (9099.25390625 MB free)
  Uptime: 540.07 sec
  Load Avg:  1.9  1.61  0.88
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Baseline result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 19 Jul 2025 - 00:25
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float32"]`               |   7.719 ms (5%) |         |        |             |
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               |   8.925 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float32"]`                                         |   3.874 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float64"]`                                         |   4.148 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float32"]`                                            |  87.698 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float64"]`                                            |  90.510 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.344 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.632 s (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  10.397 ms (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  10.117 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 391.790 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 383.237 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 136.988 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 146.860 ms (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 416.809 μs (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 415.125 μs (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 412.584 ms (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 393.152 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 568.298 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 652.276 ms (5%) |         |        |             |

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
  uname: Linux 6.11.0-1018-azure #18~24.04.1-Ubuntu SMP Sat Jun 28 04:46:03 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3224 MHz       2668 s          0 s        104 s       4774 s          0 s
       #2  3242 MHz       2750 s          0 s         99 s       4701 s          0 s
       #3  3242 MHz       2870 s          0 s        114 s       4570 s          0 s
       #4  3249 MHz       4460 s          0 s        109 s       2978 s          0 s
  Memory: 15.620681762695312 GB (13689.79296875 MB free)
  Uptime: 757.49 sec
  Load Avg:  1.68  1.59  1.03
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
    BogoMIPS:                             4890.85
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


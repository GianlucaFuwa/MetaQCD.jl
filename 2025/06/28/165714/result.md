# Benchmark result

* Pull request commit: [`9b69d08fec147f8f74e09e55024731879c2b2755`](https://github.com/GianlucaFuwa/MetaQCD.jl/commit/9b69d08fec147f8f74e09e55024731879c2b2755)
* Pull request: <https://github.com/GianlucaFuwa/MetaQCD.jl/pull/20> (Multiple timescale integrator)

# Judge result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmarks:
    - Target: 28 Jun 2025 - 16:53
    - Baseline: 28 Jun 2025 - 16:57
* Package commits:
    - Target: 70c456d
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

| ID                                                                          | time ratio     | memory ratio |
|-----------------------------------------------------------------------------|----------------|--------------|
| `["dirac", "Staggered", "Float32"]`                                         | 86.12 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Staggered", "Float64"]`                                         | 20.61 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Wilson", "Float32"]`                                            |  4.42 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Wilson", "Float64"]`                                            |  2.31 (5%) :x: | Inf (1%) :x: |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |     0.95 (5%)  | Inf (1%) :x: |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |     1.00 (5%)  | Inf (1%) :x: |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  1.44 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  1.48 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Energy Density, Float32"]`                       |  1.89 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Energy Density, Float64"]`                       |  1.97 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    |  2.00 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    |  1.88 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        |     1.04 (5%)  | Inf (1%) :x: |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        |  1.26 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      |  1.12 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      |  1.22 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             |  1.58 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             |  1.53 (5%) :x: | Inf (1%) :x: |

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
       #1  3244 MHz       1792 s          0 s        118 s      13819 s          0 s
       #2  2445 MHz       2724 s          0 s        109 s      12903 s          0 s
       #3  3248 MHz       2912 s          0 s        178 s      12648 s          0 s
       #4  3242 MHz       2126 s          0 s        142 s      13463 s          0 s
  Memory: 15.620769500732422 GB (9257.84765625 MB free)
  Uptime: 1576.02 sec
  Load Avg:  1.83  1.55  0.85
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
       #1  3243 MHz       2001 s          0 s        125 s      15760 s          0 s
       #2  2445 MHz       3662 s          0 s        116 s      14119 s          0 s
       #3  3240 MHz       3956 s          0 s        187 s      13756 s          0 s
       #4  3244 MHz       3177 s          0 s        154 s      14561 s          0 s
  Memory: 15.620769500732422 GB (13718.35546875 MB free)
  Uptime: 1792.27 sec
  Load Avg:  1.65  1.54  1.0
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Target result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 28 Jun 2025 - 16:53
* Package commit: 70c456d
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
| `["dirac", "Staggered", "Float32"]`                                         | 334.074 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Staggered", "Float64"]`                                         |  85.139 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Wilson", "Float32"]`                                            | 387.763 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Wilson", "Float64"]`                                            | 208.624 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Wilson-Clover", "Float32"]`                                     |    2.096 s (5%) |         |   8.50 KiB (1%) |           2 |
| `["dirac", "Wilson-Clover", "Float64"]`                                     |    1.804 s (5%) |         |   8.50 KiB (1%) |           2 |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.267 s (5%) |         | 127.50 KiB (1%) |          80 |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.571 s (5%) |         | 127.50 KiB (1%) |          80 |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  14.903 ms (5%) |         |   1.45 KiB (1%) |           1 |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  14.983 ms (5%) |         |   1.45 KiB (1%) |           1 |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 720.421 ms (5%) |         |   6.23 KiB (1%) |           4 |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 738.366 ms (5%) |         |   6.23 KiB (1%) |           4 |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 272.817 ms (5%) |         |  10.17 KiB (1%) |           7 |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 275.770 ms (5%) |         |  10.17 KiB (1%) |           7 |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 425.292 μs (5%) |         |   1.59 KiB (1%) |           1 |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 608.021 μs (5%) |         |   1.59 KiB (1%) |           1 |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 456.542 ms (5%) |         |   4.50 KiB (1%) |           3 |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 467.987 ms (5%) |         |   4.50 KiB (1%) |           3 |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 874.106 ms (5%) |         |  25.50 KiB (1%) |          16 |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 997.368 ms (5%) |         |  25.50 KiB (1%) |          16 |

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
       #1  3244 MHz       1792 s          0 s        118 s      13819 s          0 s
       #2  2445 MHz       2724 s          0 s        109 s      12903 s          0 s
       #3  3248 MHz       2912 s          0 s        178 s      12648 s          0 s
       #4  3242 MHz       2126 s          0 s        142 s      13463 s          0 s
  Memory: 15.620769500732422 GB (9257.84765625 MB free)
  Uptime: 1576.02 sec
  Load Avg:  1.83  1.55  0.85
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Baseline result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 28 Jun 2025 - 16:57
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float32"]`               |   7.764 ms (5%) |         |        |             |
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               |   9.184 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float32"]`                                         |   3.879 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float64"]`                                         |   4.131 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float32"]`                                            |  87.819 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float64"]`                                            |  90.462 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.331 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.565 s (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  10.380 ms (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  10.121 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 380.362 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 375.157 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 136.669 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 146.343 ms (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 408.753 μs (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 482.752 μs (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 407.055 ms (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 383.951 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 551.833 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 651.381 ms (5%) |         |        |             |

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
       #1  3243 MHz       2001 s          0 s        125 s      15760 s          0 s
       #2  2445 MHz       3662 s          0 s        116 s      14119 s          0 s
       #3  3240 MHz       3956 s          0 s        187 s      13756 s          0 s
       #4  3244 MHz       3177 s          0 s        154 s      14561 s          0 s
  Memory: 15.620769500732422 GB (13718.35546875 MB free)
  Uptime: 1792.27 sec
  Load Avg:  1.65  1.54  1.0
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


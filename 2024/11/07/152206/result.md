# Benchmark result

* Pull request commit: [`f3f4a97fa12efedf4e5d6fe6bb221fbd8c7316df`](https://github.com/GianlucaFuwa/MetaQCD.jl/commit/f3f4a97fa12efedf4e5d6fe6bb221fbd8c7316df)
* Pull request: <https://github.com/GianlucaFuwa/MetaQCD.jl/pull/17> (Feature+Refactor: Distribute fields across processes with MPI)

# Judge result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmarks:
    - Target: 7 Nov 2024 - 15:19
    - Baseline: 7 Nov 2024 - 15:21
* Package commits:
    - Target: fc167b
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
      Ubuntu 22.04.5 LTS
  uname: Linux 6.5.0-1025-azure #26~22.04.1-Ubuntu SMP Thu Jul 11 22:33:04 UTC 2024 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  2741 MHz       1546 s          0 s        122 s       3832 s          0 s
       #2  2555 MHz       2270 s          0 s         98 s       3139 s          0 s
       #3  3118 MHz       2644 s          0 s         97 s       2757 s          0 s
       #4  3242 MHz       1809 s          0 s        122 s       3570 s          0 s
  Memory: 15.606487274169922 GB (13453.41796875 MB free)
  Uptime: 553.87 sec
  Load Avg:  1.96  1.53  0.81
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
  uname: Linux 6.5.0-1025-azure #26~22.04.1-Ubuntu SMP Thu Jul 11 22:33:04 UTC 2024 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3243 MHz       2197 s          0 s        141 s       4718 s          0 s
       #2  2704 MHz       2912 s          0 s        115 s       4036 s          0 s
       #3  2445 MHz       3179 s          0 s        110 s       3766 s          0 s
       #4  2594 MHz       2206 s          0 s        149 s       4703 s          0 s
  Memory: 15.606487274169922 GB (13488.87109375 MB free)
  Uptime: 709.89 sec
  Load Avg:  1.81  1.55  0.93
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Target result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 7 Nov 2024 - 15:19
* Package commit: fc167b
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float32"]`               |   7.613 ms (5%) |         |        |             |
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               |   8.509 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float32"]`                                         |   3.854 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float64"]`                                         |   4.228 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float32"]`                                            |  87.195 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float64"]`                                            |  89.793 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.297 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.625 s (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  10.306 ms (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |   9.992 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 374.752 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 388.186 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 135.506 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 146.959 ms (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 413.429 μs (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 430.792 μs (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 405.094 ms (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 396.391 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 558.394 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 666.945 ms (5%) |         |        |             |

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
      Ubuntu 22.04.5 LTS
  uname: Linux 6.5.0-1025-azure #26~22.04.1-Ubuntu SMP Thu Jul 11 22:33:04 UTC 2024 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  2741 MHz       1546 s          0 s        122 s       3832 s          0 s
       #2  2555 MHz       2270 s          0 s         98 s       3139 s          0 s
       #3  3118 MHz       2644 s          0 s         97 s       2757 s          0 s
       #4  3242 MHz       1809 s          0 s        122 s       3570 s          0 s
  Memory: 15.606487274169922 GB (13453.41796875 MB free)
  Uptime: 553.87 sec
  Load Avg:  1.96  1.53  0.81
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Baseline result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 7 Nov 2024 - 15:21
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
| `["dirac", "MetaQCD.DiracOperators.StaggeredDiracOperator", "Float32"]`     |   3.889 ms (5%) |         |        |             |
| `["dirac", "MetaQCD.DiracOperators.StaggeredDiracOperator", "Float64"]`     |   4.378 ms (5%) |         |        |             |
| `["dirac", "MetaQCD.DiracOperators.WilsonDiracOperator", "Float32"]`        |  87.989 ms (5%) |         |        |             |
| `["dirac", "MetaQCD.DiracOperators.WilsonDiracOperator", "Float64"]`        |  92.737 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.352 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.700 s (5%) |         |        |             |
| `["meas", "measurements", "Float32, ed"]`                                   | 400.576 ms (5%) |         |        |             |
| `["meas", "measurements", "Float32, gaction"]`                              | 136.126 ms (5%) |         |        |             |
| `["meas", "measurements", "Float32, plaq"]`                                 |  10.379 ms (5%) |         |        |             |
| `["meas", "measurements", "Float32, poly"]`                                 | 566.565 μs (5%) |         |        |             |
| `["meas", "measurements", "Float32, topo"]`                                 | 422.981 ms (5%) |         |        |             |
| `["meas", "measurements", "Float32, wilson"]`                               | 556.576 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, ed"]`                                   | 400.192 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, gaction"]`                              | 153.755 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, plaq"]`                                 |  10.074 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, poly"]`                                 | 676.800 μs (5%) |         |        |             |
| `["meas", "measurements", "Float64, topo"]`                                 | 409.364 ms (5%) |         |        |             |
| `["meas", "measurements", "Float64, wilson"]`                               | 714.202 ms (5%) |         |        |             |

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
  uname: Linux 6.5.0-1025-azure #26~22.04.1-Ubuntu SMP Thu Jul 11 22:33:04 UTC 2024 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3243 MHz       2197 s          0 s        141 s       4718 s          0 s
       #2  2704 MHz       2912 s          0 s        115 s       4036 s          0 s
       #3  2445 MHz       3179 s          0 s        110 s       3766 s          0 s
       #4  2594 MHz       2206 s          0 s        149 s       4703 s          0 s
  Memory: 15.606487274169922 GB (13488.87109375 MB free)
  Uptime: 709.89 sec
  Load Avg:  1.81  1.55  0.93
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

    Architecture:                       x86_64
    CPU op-mode(s):                     32-bit, 64-bit
    Address sizes:                      48 bits physical, 48 bits virtual
    Byte Order:                         Little Endian
    CPU(s):                             4
    On-line CPU(s) list:                0-3
    Vendor ID:                          AuthenticAMD
    Model name:                         AMD EPYC 7763 64-Core Processor
    CPU family:                         25
    Model:                              1
    Thread(s) per core:                 2
    Core(s) per socket:                 2
    Socket(s):                          1
    Stepping:                           1
    BogoMIPS:                           4890.86
    Flags:                              fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl tsc_reliable nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy svm cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw topoext invpcid_single vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves clzero xsaveerptr rdpru arat npt nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload umip vaes vpclmulqdq rdpid fsrm
    Virtualization:                     AMD-V
    Hypervisor vendor:                  Microsoft
    Virtualization type:                full
    L1d cache:                          64 KiB (2 instances)
    L1i cache:                          64 KiB (2 instances)
    L2 cache:                           1 MiB (2 instances)
    L3 cache:                           32 MiB (1 instance)
    NUMA node(s):                       1
    NUMA node0 CPU(s):                  0-3
    Vulnerability Gather data sampling: Not affected
    Vulnerability Itlb multihit:        Not affected
    Vulnerability L1tf:                 Not affected
    Vulnerability Mds:                  Not affected
    Vulnerability Meltdown:             Not affected
    Vulnerability Mmio stale data:      Not affected
    Vulnerability Retbleed:             Not affected
    Vulnerability Spec rstack overflow: Vulnerable: Safe RET, no microcode
    Vulnerability Spec store bypass:    Vulnerable
    Vulnerability Spectre v1:           Mitigation; usercopy/swapgs barriers and __user pointer sanitization
    Vulnerability Spectre v2:           Mitigation; Retpolines; STIBP disabled; RSB filling; PBRSB-eIBRS Not affected; BHI Not affected
    Vulnerability Srbds:                Not affected
    Vulnerability Tsx async abort:      Not affected
    

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


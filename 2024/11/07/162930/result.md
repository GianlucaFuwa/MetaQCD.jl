# Benchmark result


# Judge result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmarks:
    - Target: 7 Nov 2024 - 16:26
    - Baseline: 7 Nov 2024 - 16:29
* Package commits:
    - Target: 5302db
    - Baseline: 5302db
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               | 0.93 (5%) :white_check_mark: |   1.00 (1%)  |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` | 0.84 (5%) :white_check_mark: |   1.00 (1%)  |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` | 0.94 (5%) :white_check_mark: |   1.00 (1%)  |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 0.94 (5%) :white_check_mark: |   1.00 (1%)  |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 0.75 (5%) :white_check_mark: |   1.00 (1%)  |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 0.95 (5%) :white_check_mark: |   1.00 (1%)  |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["dirac", "Staggered (Even-Odd preconditioned)"]`
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
      Ubuntu 22.04.5 LTS
  uname: Linux 6.5.0-1025-azure #26~22.04.1-Ubuntu SMP Thu Jul 11 22:33:04 UTC 2024 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  2575 MHz       1658 s          0 s        111 s       4041 s          0 s
       #2  2595 MHz       2228 s          0 s         92 s       3499 s          0 s
       #3  3243 MHz       2140 s          0 s        124 s       3527 s          0 s
       #4  2445 MHz       2218 s          0 s        116 s       3471 s          0 s
  Memory: 15.606491088867188 GB (13453.7421875 MB free)
  Uptime: 584.23 sec
  Load Avg:  1.93  1.48  0.76
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
       #1  2574 MHz       2297 s          0 s        121 s       4957 s          0 s
       #2  2445 MHz       2608 s          0 s        104 s       4670 s          0 s
       #3  3243 MHz       2692 s          0 s        144 s       4520 s          0 s
       #4  2445 MHz       2927 s          0 s        126 s       4317 s          0 s
  Memory: 15.606491088867188 GB (13458.10546875 MB free)
  Uptime: 741.01 sec
  Load Avg:  1.57  1.5  0.89
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Target result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 7 Nov 2024 - 16:26
* Package commit: 5302db
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float32"]`               |   7.624 ms (5%) |         |        |             |
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               |   8.479 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float32"]`                                         |   3.870 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float64"]`                                         |   4.145 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float32"]`                                            |  87.097 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float64"]`                                            |  90.940 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.268 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.591 s (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  10.299 ms (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |   9.997 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 374.059 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 376.243 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 135.537 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 144.379 ms (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 416.299 μs (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 429.524 μs (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 403.846 ms (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 405.878 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 550.378 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 652.801 ms (5%) |         |        |             |

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
       #1  2575 MHz       1658 s          0 s        111 s       4041 s          0 s
       #2  2595 MHz       2228 s          0 s         92 s       3499 s          0 s
       #3  3243 MHz       2140 s          0 s        124 s       3527 s          0 s
       #4  2445 MHz       2218 s          0 s        116 s       3471 s          0 s
  Memory: 15.606491088867188 GB (13453.7421875 MB free)
  Uptime: 584.23 sec
  Load Avg:  1.93  1.48  0.76
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-14.0.6 (ORCJIT, znver3)
  Threads: 2 on 4 virtual cores
```

---
# Baseline result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 7 Nov 2024 - 16:29
* Package commit: 5302db
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float32"]`               |   7.650 ms (5%) |         |        |             |
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               |   9.091 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float32"]`                                         |   3.914 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float64"]`                                         |   4.176 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float32"]`                                            |  88.508 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float64"]`                                            |  92.500 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.504 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.700 s (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  10.318 ms (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  10.006 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 392.382 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 393.292 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 135.680 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 152.815 ms (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 418.503 μs (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 571.640 μs (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 412.463 ms (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 403.845 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 578.647 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 689.015 ms (5%) |         |        |             |

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
       #1  2574 MHz       2297 s          0 s        121 s       4957 s          0 s
       #2  2445 MHz       2608 s          0 s        104 s       4670 s          0 s
       #3  3243 MHz       2692 s          0 s        144 s       4520 s          0 s
       #4  2445 MHz       2927 s          0 s        126 s       4317 s          0 s
  Memory: 15.606491088867188 GB (13458.10546875 MB free)
  Uptime: 741.01 sec
  Load Avg:  1.57  1.5  0.89
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


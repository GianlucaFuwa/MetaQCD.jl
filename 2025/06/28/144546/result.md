# Benchmark result

* Pull request commit: [`4d9d2006ef6cbfab8613594c9e8e9e7e96c12dec`](https://github.com/GianlucaFuwa/MetaQCD.jl/commit/4d9d2006ef6cbfab8613594c9e8e9e7e96c12dec)
* Pull request: <https://github.com/GianlucaFuwa/MetaQCD.jl/pull/20> (Multiple timescale integrator)

# Judge result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmarks:
    - Target: 28 Jun 2025 - 14:42
    - Baseline: 28 Jun 2025 - 14:45
* Package commits:
    - Target: 705e580
    - Baseline: 8a6ac11
* Julia commits:
    - Target: 760b2e5
    - Baseline: 760b2e5
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
| `["dirac", "Staggered", "Float32"]`                                         | 31.87 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Staggered", "Float64"]`                                         | 22.71 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Wilson", "Float32"]`                                            |  2.21 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Wilson", "Float64"]`                                            |  1.86 (5%) :x: | Inf (1%) :x: |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |  1.13 (5%) :x: | Inf (1%) :x: |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |     0.99 (5%)  | Inf (1%) :x: |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  1.52 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  1.54 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Energy Density, Float32"]`                       |  1.43 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Energy Density, Float64"]`                       |  1.92 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    |  1.56 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    |  1.57 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        |  1.07 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        |  1.15 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      |  1.11 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      |  1.14 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             |  1.23 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             |  1.24 (5%) :x: | Inf (1%) :x: |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["dirac", "Staggered"]`
- `["dirac", "Wilson"]`
- `["gauge", "1HB + 4OR"]`
- `["meas", "measurements"]`

## Julia versioninfo

### Target
```
Julia Version 1.11.5
Commit 760b2e5b739 (2025-04-14 06:53 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 24.04.2 LTS
  uname: Linux 6.11.0-1015-azure #15~24.04.1-Ubuntu SMP Thu May  1 02:52:08 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1     0 MHz       2016 s          0 s         95 s       3121 s          0 s
       #2     0 MHz       2747 s          0 s        103 s       2386 s          0 s
       #3     0 MHz       2462 s          0 s        125 s       2652 s          0 s
       #4     0 MHz       2615 s          0 s        124 s       2490 s          0 s
  Memory: 15.620769500732422 GB (9249.1328125 MB free)
  Uptime: 525.55 sec
  Load Avg:  1.82  1.74  1.01
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, znver3)
Threads: 2 default, 0 interactive, 1 GC (on 4 virtual cores)
```

### Baseline
```
Julia Version 1.11.5
Commit 760b2e5b739 (2025-04-14 06:53 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 24.04.2 LTS
  uname: Linux 6.11.0-1015-azure #15~24.04.1-Ubuntu SMP Thu May  1 02:52:08 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1     0 MHz       2462 s          0 s        104 s       4774 s          0 s
       #2     0 MHz       3701 s          0 s        111 s       3534 s          0 s
       #3     0 MHz       3392 s          0 s        130 s       3827 s          0 s
       #4     0 MHz       3515 s          0 s        138 s       3686 s          0 s
  Memory: 15.620769500732422 GB (13597.078125 MB free)
  Uptime: 736.67 sec
  Load Avg:  1.68  1.67  1.12
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, znver3)
Threads: 2 default, 0 interactive, 1 GC (on 4 virtual cores)
```

---
# Target result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 28 Jun 2025 - 14:42
* Package commit: 705e580
* Julia commit: 760b2e5
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
| `["dirac", "Staggered", "Float32"]`                                         | 110.453 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Staggered", "Float64"]`                                         |  89.096 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Wilson", "Float32"]`                                            | 202.634 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Wilson", "Float64"]`                                            | 171.388 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Wilson-Clover", "Float32"]`                                     |    1.738 s (5%) |         |   8.50 KiB (1%) |           2 |
| `["dirac", "Wilson-Clover", "Float64"]`                                     |    1.762 s (5%) |         |   8.50 KiB (1%) |           2 |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.543 s (5%) |         | 127.50 KiB (1%) |          80 |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.782 s (5%) |         | 127.50 KiB (1%) |          80 |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  15.761 ms (5%) |         |   1.45 KiB (1%) |           1 |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  14.987 ms (5%) |         |   1.45 KiB (1%) |           1 |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 585.843 ms (5%) |         |   6.23 KiB (1%) |           4 |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 764.726 ms (5%) |         |   6.23 KiB (1%) |           4 |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 217.743 ms (5%) |         |  10.17 KiB (1%) |           7 |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 219.805 ms (5%) |         |  10.17 KiB (1%) |           7 |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 507.567 μs (5%) |         |   1.59 KiB (1%) |           1 |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 637.450 μs (5%) |         |   1.59 KiB (1%) |           1 |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 461.959 ms (5%) |         |   4.50 KiB (1%) |           3 |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 471.597 ms (5%) |         |   4.50 KiB (1%) |           3 |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 811.016 ms (5%) |         |  25.50 KiB (1%) |          16 |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 894.554 ms (5%) |         |  25.50 KiB (1%) |          16 |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["dirac", "Staggered"]`
- `["dirac", "Wilson"]`
- `["dirac", "Wilson-Clover"]`
- `["gauge", "1HB + 4OR"]`
- `["meas", "measurements"]`

## Julia versioninfo
```
Julia Version 1.11.5
Commit 760b2e5b739 (2025-04-14 06:53 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 24.04.2 LTS
  uname: Linux 6.11.0-1015-azure #15~24.04.1-Ubuntu SMP Thu May  1 02:52:08 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1     0 MHz       2016 s          0 s         95 s       3121 s          0 s
       #2     0 MHz       2747 s          0 s        103 s       2386 s          0 s
       #3     0 MHz       2462 s          0 s        125 s       2652 s          0 s
       #4     0 MHz       2615 s          0 s        124 s       2490 s          0 s
  Memory: 15.620769500732422 GB (9249.1328125 MB free)
  Uptime: 525.55 sec
  Load Avg:  1.82  1.74  1.01
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, znver3)
Threads: 2 default, 0 interactive, 1 GC (on 4 virtual cores)
```

---
# Baseline result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 28 Jun 2025 - 14:45
* Package commit: 8a6ac11
* Julia commit: 760b2e5
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float32"]`               |   7.683 ms (5%) |         |        |             |
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               |   8.111 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float32"]`                                         |   3.466 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float64"]`                                         |   3.924 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float32"]`                                            |  91.665 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float64"]`                                            |  92.316 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.368 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.805 s (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  10.346 ms (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |   9.717 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 408.946 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 397.648 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 139.978 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 139.666 ms (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 472.592 μs (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 556.218 μs (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 416.089 ms (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 412.974 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 661.064 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 724.002 ms (5%) |         |        |             |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["dirac", "Staggered (Even-Odd preconditioned)"]`
- `["dirac", "Staggered"]`
- `["dirac", "Wilson"]`
- `["gauge", "1HB + 4OR"]`
- `["meas", "measurements"]`

## Julia versioninfo
```
Julia Version 1.11.5
Commit 760b2e5b739 (2025-04-14 06:53 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 24.04.2 LTS
  uname: Linux 6.11.0-1015-azure #15~24.04.1-Ubuntu SMP Thu May  1 02:52:08 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1     0 MHz       2462 s          0 s        104 s       4774 s          0 s
       #2     0 MHz       3701 s          0 s        111 s       3534 s          0 s
       #3     0 MHz       3392 s          0 s        130 s       3827 s          0 s
       #4     0 MHz       3515 s          0 s        138 s       3686 s          0 s
  Memory: 15.620769500732422 GB (13597.078125 MB free)
  Uptime: 736.67 sec
  Load Avg:  1.68  1.67  1.12
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, znver3)
Threads: 2 default, 0 interactive, 1 GC (on 4 virtual cores)
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


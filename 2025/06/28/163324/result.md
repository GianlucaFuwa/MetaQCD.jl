# Benchmark result

* Pull request commit: [`52d915e85f681c24b479059f3d75b7d2c9ddae54`](https://github.com/GianlucaFuwa/MetaQCD.jl/commit/52d915e85f681c24b479059f3d75b7d2c9ddae54)
* Pull request: <https://github.com/GianlucaFuwa/MetaQCD.jl/pull/20> (Multiple timescale integrator)

# Judge result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmarks:
    - Target: 28 Jun 2025 - 16:29
    - Baseline: 28 Jun 2025 - 16:33
* Package commits:
    - Target: eb94de2
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
| `["dirac", "Staggered", "Float32"]`                                         | 31.24 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Staggered", "Float64"]`                                         | 22.37 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Wilson", "Float32"]`                                            |  2.43 (5%) :x: | Inf (1%) :x: |
| `["dirac", "Wilson", "Float64"]`                                            |  1.87 (5%) :x: | Inf (1%) :x: |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |  1.10 (5%) :x: | Inf (1%) :x: |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |     1.01 (5%)  | Inf (1%) :x: |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  1.51 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  1.54 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Energy Density, Float32"]`                       |  1.43 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Energy Density, Float64"]`                       |  1.90 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    |  1.55 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    |  1.59 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        |  1.08 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        |  1.10 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      |  1.11 (5%) :x: | Inf (1%) :x: |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      |  1.13 (5%) :x: | Inf (1%) :x: |
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
       #1     0 MHz       1866 s          0 s         95 s       3312 s          0 s
       #2     0 MHz       2497 s          0 s         88 s       2694 s          0 s
       #3     0 MHz       2745 s          0 s        131 s       2405 s          0 s
       #4     0 MHz       2679 s          0 s        126 s       2471 s          0 s
  Memory: 15.620769500732422 GB (9262.91015625 MB free)
  Uptime: 529.78 sec
  Load Avg:  1.93  1.65  0.92
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
       #1     0 MHz       2225 s          0 s        106 s       5077 s          0 s
       #2     0 MHz       3439 s          0 s         95 s       3883 s          0 s
       #3     0 MHz       3682 s          0 s        142 s       3595 s          0 s
       #4     0 MHz       3706 s          0 s        135 s       3573 s          0 s
  Memory: 15.620769500732422 GB (13641.2578125 MB free)
  Uptime: 743.73 sec
  Load Avg:  1.69  1.64  1.07
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, znver3)
Threads: 2 default, 0 interactive, 1 GC (on 4 virtual cores)
```

---
# Target result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 28 Jun 2025 - 16:29
* Package commit: eb94de2
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
| `["dirac", "Staggered", "Float32"]`                                         | 108.033 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Staggered", "Float64"]`                                         |  89.007 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Wilson", "Float32"]`                                            | 223.088 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Wilson", "Float64"]`                                            | 171.182 ms (5%) |         |   4.25 KiB (1%) |           1 |
| `["dirac", "Wilson-Clover", "Float32"]`                                     |    1.767 s (5%) |         |   8.50 KiB (1%) |           2 |
| `["dirac", "Wilson-Clover", "Float64"]`                                     |    1.784 s (5%) |         |   8.50 KiB (1%) |           2 |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.506 s (5%) |         | 127.50 KiB (1%) |          80 |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.827 s (5%) |         | 127.50 KiB (1%) |          80 |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  15.605 ms (5%) |         |   1.45 KiB (1%) |           1 |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  14.964 ms (5%) |         |   1.45 KiB (1%) |           1 |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 587.452 ms (5%) |         |   6.23 KiB (1%) |           4 |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 765.643 ms (5%) |         |   6.23 KiB (1%) |           4 |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 217.723 ms (5%) |         |  10.17 KiB (1%) |           7 |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 223.696 ms (5%) |         |  10.17 KiB (1%) |           7 |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 511.510 μs (5%) |         |   1.59 KiB (1%) |           1 |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 609.004 μs (5%) |         |   1.59 KiB (1%) |           1 |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 463.100 ms (5%) |         |   4.50 KiB (1%) |           3 |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 478.071 ms (5%) |         |   4.50 KiB (1%) |           3 |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 815.049 ms (5%) |         |  25.50 KiB (1%) |          16 |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 906.104 ms (5%) |         |  25.50 KiB (1%) |          16 |

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
       #1     0 MHz       1866 s          0 s         95 s       3312 s          0 s
       #2     0 MHz       2497 s          0 s         88 s       2694 s          0 s
       #3     0 MHz       2745 s          0 s        131 s       2405 s          0 s
       #4     0 MHz       2679 s          0 s        126 s       2471 s          0 s
  Memory: 15.620769500732422 GB (9262.91015625 MB free)
  Uptime: 529.78 sec
  Load Avg:  1.93  1.65  0.92
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, znver3)
Threads: 2 default, 0 interactive, 1 GC (on 4 virtual cores)
```

---
# Baseline result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 28 Jun 2025 - 16:33
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float32"]`               |   7.610 ms (5%) |         |        |             |
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               |   8.023 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float32"]`                                         |   3.458 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float64"]`                                         |   3.980 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float32"]`                                            |  91.936 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float64"]`                                            |  91.676 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.373 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.804 s (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  10.322 ms (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |   9.726 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 409.733 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 402.359 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 140.208 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 140.290 ms (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 474.181 μs (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 551.436 μs (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 415.704 ms (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 423.987 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 665.345 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 729.837 ms (5%) |         |        |             |

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
       #1     0 MHz       2225 s          0 s        106 s       5077 s          0 s
       #2     0 MHz       3439 s          0 s         95 s       3883 s          0 s
       #3     0 MHz       3682 s          0 s        142 s       3595 s          0 s
       #4     0 MHz       3706 s          0 s        135 s       3573 s          0 s
  Memory: 15.620769500732422 GB (13641.2578125 MB free)
  Uptime: 743.73 sec
  Load Avg:  1.69  1.64  1.07
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


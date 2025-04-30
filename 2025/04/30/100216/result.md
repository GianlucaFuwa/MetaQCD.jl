# Benchmark result


# Judge result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmarks:
    - Target: 30 Apr 2025 - 09:58
    - Baseline: 30 Apr 2025 - 10:02
* Package commits:
    - Target: 8a6ac11
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

| ID                                                                          | time ratio | memory ratio |
|-----------------------------------------------------------------------------|------------|--------------|

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
Julia Version 1.11.5
Commit 760b2e5b739 (2025-04-14 06:53 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 24.04.2 LTS
  uname: Linux 6.11.0-1012-azure #12~24.04.1-Ubuntu SMP Mon Mar 10 19:00:39 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1     0 MHz       2187 s          0 s        147 s       4927 s          0 s
       #2     0 MHz       2489 s          0 s        126 s       4710 s          0 s
       #3     0 MHz       2040 s          0 s        131 s       5131 s          0 s
       #4     0 MHz       3090 s          0 s        125 s       4079 s          0 s
  Memory: 15.614944458007812 GB (13466.62109375 MB free)
  Uptime: 738.96 sec
  Load Avg:  1.91  1.64  0.91
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
  uname: Linux 6.11.0-1012-azure #12~24.04.1-Ubuntu SMP Mon Mar 10 19:00:39 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1     0 MHz       2841 s          0 s        164 s       6200 s          0 s
       #2     0 MHz       3273 s          0 s        140 s       5857 s          0 s
       #3     0 MHz       2820 s          0 s        150 s       6278 s          0 s
       #4     0 MHz       3935 s          0 s        144 s       5161 s          0 s
  Memory: 15.614944458007812 GB (13498.0625 MB free)
  Uptime: 933.69 sec
  Load Avg:  1.77  1.65  1.05
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, znver3)
Threads: 2 default, 0 interactive, 1 GC (on 4 virtual cores)
```

---
# Target result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 30 Apr 2025 - 09:58
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float32"]`               |   7.614 ms (5%) |         |        |             |
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               |   7.902 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float32"]`                                         |   3.462 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float64"]`                                         |   3.984 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float32"]`                                            |  91.225 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float64"]`                                            |  92.599 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.395 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.817 s (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  10.334 ms (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |   9.730 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 418.774 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 409.662 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 146.626 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 145.124 ms (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 473.210 μs (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 561.154 μs (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 416.800 ms (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 433.020 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 665.783 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 738.898 ms (5%) |         |        |             |

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
  uname: Linux 6.11.0-1012-azure #12~24.04.1-Ubuntu SMP Mon Mar 10 19:00:39 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1     0 MHz       2187 s          0 s        147 s       4927 s          0 s
       #2     0 MHz       2489 s          0 s        126 s       4710 s          0 s
       #3     0 MHz       2040 s          0 s        131 s       5131 s          0 s
       #4     0 MHz       3090 s          0 s        125 s       4079 s          0 s
  Memory: 15.614944458007812 GB (13466.62109375 MB free)
  Uptime: 738.96 sec
  Load Avg:  1.91  1.64  0.91
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, znver3)
Threads: 2 default, 0 interactive, 1 GC (on 4 virtual cores)
```

---
# Baseline result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 30 Apr 2025 - 10:02
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float32"]`               |   7.606 ms (5%) |         |        |             |
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               |   8.161 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float32"]`                                         |   3.469 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float64"]`                                         |   4.014 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float32"]`                                            |  91.527 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float64"]`                                            |  91.492 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.459 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.869 s (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  10.257 ms (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |   9.735 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 410.989 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 399.460 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 139.710 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 140.537 ms (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 468.201 μs (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 565.954 μs (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 413.953 ms (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 430.255 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 677.732 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 729.972 ms (5%) |         |        |             |

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
  uname: Linux 6.11.0-1012-azure #12~24.04.1-Ubuntu SMP Mon Mar 10 19:00:39 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1     0 MHz       2841 s          0 s        164 s       6200 s          0 s
       #2     0 MHz       3273 s          0 s        140 s       5857 s          0 s
       #3     0 MHz       2820 s          0 s        150 s       6278 s          0 s
       #4     0 MHz       3935 s          0 s        144 s       5161 s          0 s
  Memory: 15.614944458007812 GB (13498.0625 MB free)
  Uptime: 933.69 sec
  Load Avg:  1.77  1.65  1.05
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


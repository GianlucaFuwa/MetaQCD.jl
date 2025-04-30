# Benchmark result


# Judge result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmarks:
    - Target: 30 Apr 2025 - 09:58
    - Baseline: 30 Apr 2025 - 10:01
* Package commits:
    - Target: 8a6ac11
    - Baseline: 8a6ac11
* Julia commits:
    - Target: 5595d20
    - Baseline: 5595d20
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               | 0.84 (5%) :white_check_mark: |   1.00 (1%)  |

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
Julia Version 1.10.9
Commit 5595d20a287 (2025-03-10 12:51 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 24.04.2 LTS
  uname: Linux 6.11.0-1012-azure #12~24.04.1-Ubuntu SMP Mon Mar 10 19:00:39 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3243 MHz       1983 s          0 s        136 s       4242 s          0 s
       #2  3252 MHz       2106 s          0 s        106 s       4146 s          0 s
       #3  3240 MHz       1753 s          0 s        129 s       4462 s          0 s
       #4  3244 MHz       2927 s          0 s        121 s       3320 s          0 s
  Memory: 15.614940643310547 GB (13560.23828125 MB free)
  Uptime: 644.98 sec
  Load Avg:  1.98  1.6  0.84
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver3)
Threads: 2 default, 0 interactive, 1 GC (on 4 virtual cores)
```

### Baseline
```
Julia Version 1.10.9
Commit 5595d20a287 (2025-03-10 12:51 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 24.04.2 LTS
  uname: Linux 6.11.0-1012-azure #12~24.04.1-Ubuntu SMP Mon Mar 10 19:00:39 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3231 MHz       2735 s          0 s        153 s       5366 s          0 s
       #2  3243 MHz       2805 s          0 s        120 s       5327 s          0 s
       #3  3262 MHz       2172 s          0 s        145 s       5920 s          0 s
       #4  3240 MHz       4067 s          0 s        137 s       4059 s          0 s
  Memory: 15.614940643310547 GB (13513.35546875 MB free)
  Uptime: 834.61 sec
  Load Avg:  1.75  1.61  0.99
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver3)
Threads: 2 default, 0 interactive, 1 GC (on 4 virtual cores)
```

---
# Target result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 30 Apr 2025 - 09:58
* Package commit: 8a6ac11
* Julia commit: 5595d20
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float32"]`               |   7.542 ms (5%) |         |        |             |
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               |   8.240 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float32"]`                                         |   3.560 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float64"]`                                         |   4.216 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float32"]`                                            |  84.301 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float64"]`                                            |  84.373 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.354 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.705 s (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  10.483 ms (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  10.154 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 371.869 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 367.939 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 146.104 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 146.514 ms (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 399.021 μs (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 489.620 μs (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 373.168 ms (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 363.905 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 558.495 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 668.478 ms (5%) |         |        |             |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["dirac", "Staggered (Even-Odd preconditioned)"]`
- `["dirac", "Staggered"]`
- `["dirac", "Wilson"]`
- `["gauge", "1HB + 4OR"]`
- `["meas", "measurements"]`

## Julia versioninfo
```
Julia Version 1.10.9
Commit 5595d20a287 (2025-03-10 12:51 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 24.04.2 LTS
  uname: Linux 6.11.0-1012-azure #12~24.04.1-Ubuntu SMP Mon Mar 10 19:00:39 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3243 MHz       1983 s          0 s        136 s       4242 s          0 s
       #2  3252 MHz       2106 s          0 s        106 s       4146 s          0 s
       #3  3240 MHz       1753 s          0 s        129 s       4462 s          0 s
       #4  3244 MHz       2927 s          0 s        121 s       3320 s          0 s
  Memory: 15.614940643310547 GB (13560.23828125 MB free)
  Uptime: 644.98 sec
  Load Avg:  1.98  1.6  0.84
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver3)
Threads: 2 default, 0 interactive, 1 GC (on 4 virtual cores)
```

---
# Baseline result
# Benchmark Report for */home/runner/work/MetaQCD.jl/MetaQCD.jl*

## Job Properties
* Time of benchmark: 30 Apr 2025 - 10:01
* Package commit: 8a6ac11
* Julia commit: 5595d20
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
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float32"]`               |   7.596 ms (5%) |         |        |             |
| `["dirac", "Staggered (Even-Odd preconditioned)", "Float64"]`               |   9.852 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float32"]`                                         |   3.565 ms (5%) |         |        |             |
| `["dirac", "Staggered", "Float64"]`                                         |   4.224 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float32"]`                                            |  83.784 ms (5%) |         |        |             |
| `["dirac", "Wilson", "Float64"]`                                            |  86.467 ms (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float32, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.310 s (5%) |         |        |             |
| `["gauge", "1HB + 4OR", "Float64, MetaQCD.Fields.SymanzikTreeGaugeAction"]` |    1.648 s (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float32"]`                        |  10.466 ms (5%) |         |        |             |
| `["meas", "measurements", "Avg Plaquette, Float64"]`                        |  10.117 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float32"]`                       | 368.029 ms (5%) |         |        |             |
| `["meas", "measurements", "Energy Density, Float64"]`                       | 361.313 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float32"]`    | 145.964 ms (5%) |         |        |             |
| `["meas", "measurements", "Gauge Action (W + LW + IW + DBW2), Float64"]`    | 146.531 ms (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float32"]`                        | 407.668 μs (5%) |         |        |             |
| `["meas", "measurements", "Polyakov Loop, Float64"]`                        | 489.601 μs (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float32"]`      | 372.413 ms (5%) |         |        |             |
| `["meas", "measurements", "Top. Charge (Plaq + Clov + Imp), Float64"]`      | 371.420 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float32"]`             | 541.879 ms (5%) |         |        |             |
| `["meas", "measurements", "Wilson Loops (2x2 + 4x4), Float64"]`             | 650.930 ms (5%) |         |        |             |

## Benchmark Group List
Here's a list of all the benchmark groups executed by this job:

- `["dirac", "Staggered (Even-Odd preconditioned)"]`
- `["dirac", "Staggered"]`
- `["dirac", "Wilson"]`
- `["gauge", "1HB + 4OR"]`
- `["meas", "measurements"]`

## Julia versioninfo
```
Julia Version 1.10.9
Commit 5595d20a287 (2025-03-10 12:51 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
      Ubuntu 24.04.2 LTS
  uname: Linux 6.11.0-1012-azure #12~24.04.1-Ubuntu SMP Mon Mar 10 19:00:39 UTC 2025 x86_64 x86_64
  CPU: AMD EPYC 7763 64-Core Processor: 
              speed         user         nice          sys         idle          irq
       #1  3231 MHz       2735 s          0 s        153 s       5366 s          0 s
       #2  3243 MHz       2805 s          0 s        120 s       5327 s          0 s
       #3  3262 MHz       2172 s          0 s        145 s       5920 s          0 s
       #4  3240 MHz       4067 s          0 s        137 s       4059 s          0 s
  Memory: 15.614940643310547 GB (13513.35546875 MB free)
  Uptime: 834.61 sec
  Load Avg:  1.75  1.61  0.99
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver3)
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
    BogoMIPS:                             4890.86
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


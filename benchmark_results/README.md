# vLLM Runtime Benchmark: Base vs RunAI Sharded

## Scope
- Compare vLLM initialization/runtime loading behavior for:
- Base model: `models/gpt-oss-120b` (default load path)
- RunAI sharded model: `models/gpt-oss-120b-runai-sharded-tp2` with `load_format=runai_streamer_sharded`
- Both runs use `tensor_parallel_size=2`, `dtype=auto`, `gpu_memory_utilization=0.92`, `max_model_len=4096`.

## Visualization
![vLLM load benchmark](./vllm_load_benchmark.png)

## Key Metrics
| Metric | Base | RunAI sharded |
|---|---:|---:|
| Wall time to model ready (s) | 421.89 | 62.25 |
| Weights loading time (s) | 387.56 | 28.75 |
| Model loading time (s) | 390.37 | 31.53 |
| Engine init time (s) | 22.52 | 22.70 |
| torch.compile total (s) | 8.94 | 8.95 |
| Graph capture time (s) | 10.00 | 10.00 |
| Model loading memory (GiB) | 34.38 | 34.38 |
| Graph capture memory (GiB) | 1.01 | 1.01 |

## Speedups (Base / RunAI)
| Metric | Speedup |
|---|---:|
| Wall time to ready | 6.78x |
| Weights loading | 13.48x |
| Model loading | 12.38x |

## RunAI Streamer Metrics
- Streamed size: 34.30 GiB
- Best observed stream time: 28.52 s
- Best observed throughput: 1.20 GiB/s

## Relevant vLLM Initialization Log Excerpts

### Base
```text
(EngineCore_DP0 pid=19567) (Worker_TP0 pid=19573) INFO 02-15 05:15:23 [default_loader.py:308] Loading weights took 387.56 seconds
(EngineCore_DP0 pid=19567) (Worker_TP0 pid=19573) INFO 02-15 05:15:26 [gpu_model_runner.py:3549] Model loading took 34.3774 GiB memory and 390.373224 seconds
(EngineCore_DP0 pid=19567) (Worker_TP0 pid=19573) INFO 02-15 05:15:37 [monitor.py:34] torch.compile takes 8.94 s in total
(EngineCore_DP0 pid=19567) (Worker_TP0 pid=19573) INFO 02-15 05:15:49 [gpu_model_runner.py:4466] Graph capturing finished in 10 secs, took 1.01 GiB
(EngineCore_DP0 pid=19567) INFO 02-15 05:15:49 [core.py:254] init engine (profile, create kv cache, warmup model) took 22.52 seconds
BENCHMARK_READY_SECONDS=421.8885
```

### RunAI sharded
```text
(EngineCore_DP0 pid=20129) (Worker_TP1 pid=20137) [2026-02-15 05:16:38] INFO file_streamer.py:66: [RunAI Streamer] Overall time to stream 34.3 GiB of all files to cpu: 28.52s, 1.2 GiB/s
(EngineCore_DP0 pid=20129) (Worker_TP0 pid=20135) [2026-02-15 05:16:38] INFO file_streamer.py:66: [RunAI Streamer] Overall time to stream 34.3 GiB of all files to cpu: 28.74s, 1.2 GiB/s
(EngineCore_DP0 pid=20129) (Worker_TP0 pid=20135) INFO 02-15 05:16:38 [run_lm_eval_runai_sharded.py:150] Loading weights took 28.75 seconds
(EngineCore_DP0 pid=20129) (Worker_TP0 pid=20135) INFO 02-15 05:16:39 [gpu_model_runner.py:3549] Model loading took 34.3774 GiB memory and 31.534663 seconds
(EngineCore_DP0 pid=20129) (Worker_TP0 pid=20135) INFO 02-15 05:16:50 [monitor.py:34] torch.compile takes 8.95 s in total
(EngineCore_DP0 pid=20129) (Worker_TP0 pid=20135) INFO 02-15 05:17:02 [gpu_model_runner.py:4466] Graph capturing finished in 10 secs, took 1.01 GiB
(EngineCore_DP0 pid=20129) INFO 02-15 05:17:02 [core.py:254] init engine (profile, create kv cache, warmup model) took 22.70 seconds
BENCHMARK_READY_SECONDS=62.2471
```

## Raw Artifacts
- Base log: `benchmark_results/base.log`
- RunAI log: `benchmark_results/runai_sharded.log`
- Base excerpts: `benchmark_results/base_excerpts.log`
- RunAI excerpts: `benchmark_results/runai_sharded_excerpts.log`
- Metrics JSON: `benchmark_results/benchmark_metrics.json`

# LM Eval Results

## Setup
- Date: 2026-02-15
- Evaluator: `lm_eval 0.4.11`
- Backend: `vllm`
- Shared model args:
  - `tensor_parallel_size=2`
  - `dtype=auto`
  - `gpu_memory_utilization=0.92`
  - `max_model_len=4096`
- Tasks: `arc_easy,piqa`
- Limit: `20` (quick comparison run)
- Batch size: `auto`
- Seed: default lm-eval seeds (`0/1234/1234/1234`)

## Commands

### Base model (no RunAI streamer)
```bash
python -m lm_eval \
  --model vllm \
  --model_args pretrained=models/gpt-oss-120b,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.92,max_model_len=4096 \
  --tasks arc_easy,piqa \
  --limit 20 \
  --batch_size auto \
  --output_path lm_eval_base
```

### RunAI-sharded model (`runai_streamer_sharded`)
```bash
python run_lm_eval_runai_sharded.py \
  --model vllm \
  --model_args pretrained=models/gpt-oss-120b-runai-sharded-tp2,load_format=runai_streamer_sharded,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.92,max_model_len=4096 \
  --tasks arc_easy,piqa \
  --limit 20 \
  --batch_size auto \
  --output_path lm_eval_runai_sharded
```

## Results

### Aggregated metrics

| Task | Metric | Base model | RunAI-sharded |
|---|---|---:|---:|
| arc_easy | acc | 0.8000 | 0.8000 |
| arc_easy | acc_norm | 0.8000 | 0.8000 |
| piqa | acc | 0.7500 | 0.7500 |
| piqa | acc_norm | 0.8500 | 0.8500 |

### lm-eval stderr (reported by lm-eval)

| Task | Metric | Base stderr | RunAI-sharded stderr |
|---|---|---:|---:|
| arc_easy | acc | 0.0918 | 0.0918 |
| arc_easy | acc_norm | 0.0918 | 0.0918 |
| piqa | acc | 0.0993 | 0.0993 |
| piqa | acc_norm | 0.0819 | 0.0819 |

### Result artifacts

- Base model JSON: `lm_eval_base/results_2026-02-15T04-48-00.777972.json`
- RunAI-sharded JSON: `lm_eval_runai_sharded/results_2026-02-15T04-54-25.512642.json`

### Notes

- This was a quick comparison run with `--limit 20`, so results are directional rather than full-benchmark quality.
- `runai_streamer_sharded` was evaluated via `run_lm_eval_runai_sharded.py`, which applies the same runtime loader patches used for successful inference in this workspace.

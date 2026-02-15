from time import perf_counter

from run_lm_eval_runai_sharded import (
    patch_marlin_load_order,
    patch_runai_dtype,
    patch_sharded_loader,
)

patch_runai_dtype()
patch_marlin_load_order()
patch_sharded_loader()

from vllm import LLM

MODEL_PATH = "/workspace/models/gpt-oss-120b-runai-sharded-tp2"

start = perf_counter()
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=2,
    load_format="runai_streamer_sharded",
    dtype="auto",
    gpu_memory_utilization=0.92,
    max_model_len=4096,
)
ready = perf_counter() - start
print(f"BENCHMARK_READY_SECONDS={ready:.4f}", flush=True)

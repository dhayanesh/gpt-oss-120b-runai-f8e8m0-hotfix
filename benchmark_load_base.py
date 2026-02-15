from time import perf_counter

from vllm import LLM

MODEL_PATH = "/workspace/models/gpt-oss-120b"

start = perf_counter()
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=2,
    dtype="auto",
    gpu_memory_utilization=0.92,
    max_model_len=4096,
)
ready = perf_counter() - start
print(f"BENCHMARK_READY_SECONDS={ready:.4f}", flush=True)

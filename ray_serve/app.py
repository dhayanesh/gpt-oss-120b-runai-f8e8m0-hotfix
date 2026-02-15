from pathlib import Path

from ray.serve.llm import LLMConfig, build_openai_app

_HOTFIX_PATH = str(Path(__file__).resolve().parent)
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="gpt-oss-120b-runai-sharded",
        model_source=f"{_PROJECT_ROOT}/models/gpt-oss-120b-runai-sharded-tp2",
        tokenizer_source=f"{_PROJECT_ROOT}/models/gpt-oss-120b-runai-sharded-tp2",
    ),
    llm_engine="vLLM",
    engine_kwargs=dict(
        tensor_parallel_size=2,
        load_format="runai_streamer_sharded",
        dtype="auto",
        gpu_memory_utilization=0.92,
        max_model_len=4096,
    ),
    runtime_env=dict(
        env_vars={
            "PYTHONPATH": f"{_PROJECT_ROOT}:{_HOTFIX_PATH}",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
    ),
)

app = build_openai_app({"llm_configs": [llm_config]})

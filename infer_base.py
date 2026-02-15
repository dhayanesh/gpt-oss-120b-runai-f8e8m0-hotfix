from vllm import LLM, SamplingParams

MODEL_PATH = "models/gpt-oss-120b"

print("Loading model from:", MODEL_PATH)

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=2,
    dtype="auto",
    gpu_memory_utilization=0.92,
    max_model_len=8192,
)

print("Model loaded successfully")

# ============================================================
# Run inference
# ============================================================

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1000,
)

messages = [
    {"role": "user", "content": "Explain the solar system in 5 concise bullet points."}
]

outputs = llm.chat(messages, sampling_params=sampling_params)

print("\n================ OUTPUT ================\n")

for output in outputs:
    print(output.outputs[0].text)
